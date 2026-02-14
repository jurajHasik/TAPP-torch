from itertools import accumulate
import operator
from typing import Union
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from torch.testing._internal.optests import opcheck

DTYPE_OPTIONS= [torch.float32, torch.float64, torch.complex64, torch.complex128]

# def reference_tensor_product(a, b, c, d,
#                              modes_a, modes_b, modes_c, modes_d,
#                              alpha, beta):
#     d= alpha * torch.einsum(a, modes_a, b, modes_b, modes_d) + (beta * c if c is not None else 0) 
#     return d

# def reference_tensordot(a, b, contract_idx_a, contract_idx_b, out_order= None):
#     res= torch.tensordot(a, b, dims=(contract_idx_a, contract_idx_b))
#     res= res.permute(out_order) if out_order is not None else res
#     return res

import tapp_torch
import json
import numpy as np
import symmray as sr
import os


class SymmrayAdapter:
    """
    Thin interface for converting between symmray AbelianArray with data backed by dict of blocks 
    and flat torch arrays, where blocks are serialized in to contiguous 1D array.
    """

    @staticmethod
    def to_torch_(T: sr.AbelianArray, device=None) -> sr.AbelianArray:
        """
        Convert data tensors to torch.Tensor in-place.
        """
        for b_idx,b in T.blocks.items():
            if isinstance(b, torch.Tensor):
                continue
            elif isinstance(b, np.ndarray):
                T.blocks[b_idx]= torch.from_numpy(b).to(device="cpu" if device is None else device)
            else:
                T.blocks[b_idx]= torch.tensor(b)
                if device is not None:
                    T.blocks[b_idx]= T.blocks[b_idx].to(device=device)
        return T

    @staticmethod
    def flatten(T: sr.AbelianArray, device=None) -> tuple:
        """Flatten a symmray AbelianArray into a 1D torch tensor.

        Creates a 1D torch data tensor from an AbelianArray by flattening the blocks
        in the order of T.blocks and concatenating them.

        Args:
            T: symmray AbelianArray to flatten
            device: torch device to place the tensor on

        Returns:
            Tuple of:
            - flat: 1D torch tensor containing all block data concatenated
            - slices: List of tuples containing (offset_tuple, shape, size) for each block
        """
        blocks = list(T.blocks.values())
        sizes = [b.numel() for b in blocks] if isinstance(blocks[0], torch.Tensor) else [b.size for b in blocks] 
        offsets = list(accumulate([0] + sizes[:-1]))  # [0, size[0], size[0]+size[1], ...]
        slices = [
            (((offset, offset + size),), block.shape, size)
            for offset, size, block in zip(offsets, sizes, blocks)
        ]
        flat = torch.cat([b.reshape(-1) for b in blocks]).to(device=device)
        return flat, slices

    @staticmethod
    def fill(flat_t: torch.Tensor, base: sr.AbelianArray) -> sr.AbelianArray:
        """Fill a symmray AbelianArray from a flattened torch tensor.

        Takes a 1D torch tensor and fills it into a symmray AbelianArray structure,
        reshaping blocks according to the base array's block structure.

        Args:
            flat_t: 1D array containing flattened block data
            base: symmray AbelianArray providing the structure (block shapes)

        Returns:
            New AbelianArray with data from flat_t in the structure of base
        """
        offset = 0
        R= base.copy()
        for b_idx, b in R.blocks.items():
            size= b.numel() if isinstance(b,torch.Tensor) else b.size
            R.blocks[b_idx] = flat_t[offset:offset + size].reshape(b.shape)
            offset += size
        return R


def make_sample_tensor_product_bs(
        A: sr.AbelianArray, B: sr.AbelianArray, C: Union[sr.AbelianArray,None], D: sr.AbelianArray, 
        a_modes, b_modes, c_modes, d_modes,
        alpha, beta, device, requires_grad=False):
    
    def get_struct_T(T: sr.AbelianArray):
        blocks= list(T.blocks.keys())
        blocks_shape= list([ b.shape for b in T.blocks.values() ])
        return {"t": blocks, "D": blocks_shape}

    test_case= {
        "modes_A": a_modes,
        "modes_B": b_modes,
        "modes_C": c_modes if C is not None else None,
        "modes_D": d_modes,
        "A": {"struct": get_struct_T(A)},
        "B": {"struct": get_struct_T(B)},
        "C": {"struct": get_struct_T(C)} if C is not None else None,
        "D": {"struct": get_struct_T(D)},
    }

    # create 1D torch data tensors from A, B, C, D by flattening the blocks in the order of T.blocks and concatenating them.
    a, test_case["A"]["slices"]= SymmrayAdapter.flatten(A, device)
    b, test_case["B"]["slices"]= SymmrayAdapter.flatten(B, device)
    c= None
    if C is not None:
        c, test_case["C"]["slices"]= SymmrayAdapter.flatten(C, device)
        c.requires_grad_(requires_grad)
    d, test_case["D"]["slices"]= SymmrayAdapter.flatten(D, device)
    a.requires_grad_(requires_grad)
    b.requires_grad_(requires_grad)
    d.mul_(0) # zero out D to ensure correctness of the test

    alpha_t= torch.tensor(alpha, dtype=d.dtype, device=device)
    beta_t= torch.tensor(beta, dtype=d.dtype, device=device)

    test_case= preprocess_structure(test_case)

    def get_tapp_torch_args_T(T):
        return ( test_case["modes_"+T], list([len(s) for s in test_case[T]["sectionsPerMode_union"]]), \
                sum(test_case[T]["extentsPerMode_union"],[]), test_case[T]["blocks_shifted_lin"], \
                test_case[T]["strides_lin"], test_case[T]["offsets"] )
    
    return (a,b,c,d) + get_tapp_torch_args_T("A") + get_tapp_torch_args_T("B") \
        + ((None,)*6 if C is None else get_tapp_torch_args_T("C")) + get_tapp_torch_args_T("D") \
        + (alpha_t, beta_t)


def make_sample_tensordot_bs(
        A: sr.AbelianArray, B: sr.AbelianArray, D: sr.AbelianArray,
        contracted_modes_A, contracted_modes_B, modes_out, device, requires_grad=False):

    # Determine the output shape and reindex modes to match the tensor_product api
    modes_A= list(range(A.ndim))
    modes_B= [modes_A[contracted_modes_A[contracted_modes_B.index(n)]] if n in contracted_modes_B else j 
              for n,j in enumerate(range(A.ndim, A.ndim+B.ndim))]
    remaining_modes_A = [i for i in modes_A if i not in contracted_modes_A]
    remaining_modes_B = [j for n,j in enumerate(modes_B) if n not in contracted_modes_B]

    modes_D= remaining_modes_A + remaining_modes_B
    if modes_out is not None:
        assert len(modes_out)==len(modes_D), "modes_out must have the same length as the number of remaining modes"
        modes_D = [modes_D[i] for i in modes_out]

    res= make_sample_tensor_product_bs(A, B, None, D, 
        modes_A, modes_B, None, modes_D,
        1, 0, device, requires_grad=requires_grad)
    
    # 0:3 : a,b,c,d
    # 4:4+6: modes_A, numSectionsPerMode_A, sectionExtents_A, blocks_A, strides_A, offsets_A
    # 10:10+6: modes_B, numSectionsPerMode_B, sectionExtents_B, blocks_B, strides_B, offsets_B
    # 16:16+6: modes_C, numSectionsPerMode_C, sectionExtents_C, blocks_C, strides_C, offsets_C
    # 22:22+6: modes_D, numSectionsPerMode_D, sectionExtents_D, blocks_D, strides_D, offsets_D
    return res[:2] + (contracted_modes_A, contracted_modes_B) \
        + res[5:5+5] + res[11:11+5] + res[23:23+5] + (modes_out,)


def preprocess_structure(case):    
    a_modes= case["modes_A"]
    b_modes= case["modes_B"]
    d_modes= case["modes_D"]
    c_modes= None if case["C"] is None else case["modes_C"]

    def sort_and_group_by_first_column(X, Y):
        result = []
        for i in range(X.shape[1]):
            x_col = X[:, i]
            y_col = Y[:, i]
            # Sort by x_col
            order = np.argsort(x_col)
            x_sorted = x_col[order]
            y_sorted = y_col[order]
            # Get unique x values and their first indices
            _, idx = np.unique(x_sorted, return_index=True)
            # Collect corresponding y values
            result.append(y_sorted[idx].tolist())
        return result

    # t holds application-specific section indices associated with non-zero blocks
    # D holds extents of blocks 
    def parse_block_structure(T: dict):
        """
        Returns:

        - section_idx_per_block: 2D array of shape (num_blocks, num_modes) where each entry is the section index of the respective mode for that block
        - sections_per_mode: list of lists, where each inner list contains the unique section indices for the respective mode, sorted in ascending order
        - extents_per_mode: list of lists, where each inner list contains the extents of the blocks corresponding to the unique section indices for the
            respective mode, sorted in the same order as sections_per_mode
        """
        section_idx_per_block= np.array(T["struct"]["t"])
        section_extents_per_block= np.array(T["struct"]["D"])
        sections_per_mode= [np.array(list(sorted(set(section_idx_per_block[:, i])))) for i in range(section_idx_per_block.shape[1])]
        sections_per_mode= [s_m if s_m.ndim==2 else s_m[..., np.newaxis] for s_m in sections_per_mode]
        extents_per_mode= sort_and_group_by_first_column(section_idx_per_block, section_extents_per_block)
        return section_idx_per_block, sections_per_mode, extents_per_mode

    def merge_sorted_lists(a, b):
        result, origin = [], [] # origin tracks whether the element came from a or b and the index
        i, j = 0, 0
        while i < len(a) and j < len(b):
            if a[i] < b[j]:
                result.append(a[i])
                origin.append((0,i)) # from a or both
                i += 1
            elif a[i] > b[j]:
                result.append(b[j])
                origin.append((1,j)) # from b
                j += 1
            else:
                result.append(a[i])
                origin.append((0,i)) # from a or both
                i += 1
                j += 1
        result.extend(a[i:])
        result.extend(b[j:])
        return result, origin

    sidxpb_A, spm_A, epm_A= parse_block_structure(case["A"])
    sidxpb_B, spm_B, epm_B= parse_block_structure(case["B"])
    sidxpb_D, spm_D, epm_D= parse_block_structure(case["D"])
    sidxpb_C, spm_C, epm_C= (None, None, None) if case["C"] is None else parse_block_structure(case["C"]) 

    # I. Relabel section indices to be 0-indexed integers
    # I.1 Find pairs of indices (i, j) where a_modes[i] == b_modes[j]
    matching_indices = [(i, j) for i, a in enumerate(a_modes) for j, b in enumerate(b_modes) if a == b]

    # I.2 For each pair, merge the section lists and create a mapping from old to new indices
    for ia, ib in matching_indices:
        merged_sections, origin = merge_sorted_lists(spm_A[ia], spm_B[ib])
        spm_A[ia] = merged_sections
        spm_B[ib] = merged_sections

        # I.3 merge extents as well
        epm_AB = [ epm_A[ia][idx] if a_or_b==0 else epm_B[ib][idx] for a_or_b,idx in origin ]
        epm_A[ia] = epm_AB
        epm_B[ib] = epm_AB

    # II. Section per mode and their extents including. For contracted modes, 
    #     a union of section indices across A and B is taken, and extents are merged accordingly.
    case["A"]["sectionsPerMode_union"] = spm_A
    case["B"]["sectionsPerMode_union"] = spm_B
    case["D"]["sectionsPerMode_union"] = spm_D
    if case["C"] is not None:
        case["C"]["sectionsPerMode_union"] = spm_C
    case["A"]["extentsPerMode_union"] = epm_A
    case["B"]["extentsPerMode_union"] = epm_B
    case["D"]["extentsPerMode_union"] = epm_D
    if case["C"] is not None:
        case["C"]["extentsPerMode_union"] = epm_C

    def compute_offsets_and_strides(T):
        slices= T["slices"]
        S= np.empty( (len(slices),len(slices[0][1])), dtype=np.int64 )
        S[:,-1]=1
        Ds= np.asarray( tuple(b[1] for b in slices) )
        np.cumprod( Ds[:, -1:0:-1], axis=-1, out= S[:,:len(slices[0][1])-1][:,::-1])
        return tuple(s[0][0][0] for s in slices), S.reshape(-1).tolist()

    # III. Compute linearized storage offsets and strides for each block
    #      strides are linearized (flattened) 
    case["A"]["offsets"], case["A"]["strides_lin"]= compute_offsets_and_strides(case["A"])
    case["B"]["offsets"], case["B"]["strides_lin"]= compute_offsets_and_strides(case["B"])
    case["D"]["offsets"], case["D"]["strides_lin"]= compute_offsets_and_strides(case["D"])
    if case["C"] is not None:
        case["C"]["offsets"], case["C"]["strides_lin"]= compute_offsets_and_strides(case["C"])
    
    # Linearization of block coordinates
    # NOT APPLICABLE: Block index can have value at most equal to number of extents in the respective mode - 1
    def normalize_section_idxs(section_idxs_per_block):
        res= []
        for t_mode in section_idxs_per_block:
            tm= np.asarray(t_mode)
            floor= np.min(tm, axis=0)

            idx2i= np.empty( np.max(tm,axis=0)-floor+1, dtype=np.int64 )
            for i, idx in enumerate(tm):
                idx2i[tuple(idx - floor)]= i

            res.append( (floor, list(accumulate( np.max(tm, axis=0)-floor+1, operator.mul )), idx2i) )
        return res
    
    def _blocksparse_coords_v3(struct_t, filled_t_per_mode):
        ts= np.array(struct_t).reshape( len(struct_t), len(filled_t_per_mode), 1 )
        n= normalize_section_idxs(filled_t_per_mode)
        ts-= np.stack([f[0] for f in n])                # shift to zero-based [:,...] -= [...] broadcast over :
        # ts[...,1:]*= np.stack([f[1] for f in n])[:,:-1] # raise by base
        # ts= np.sum(ts,axis=-1).tolist()              # compute linearized indices
        B= np.empty( ts.shape[:2], dtype=np.int64 )
        for mode in range(len(filled_t_per_mode)):
            B[:,mode]= n[mode][2][ tuple( ts[:,mode,i] for i in range(1) ) ]

        return B.reshape(-1).tolist()

    # IV. Compute shifted block coordinates for each block, which start at 0 for each mode.
    #     block coordinates are linearized (flattened)
    case["A"]["blocks_shifted_lin"] = _blocksparse_coords_v3(sidxpb_A, spm_A)
    case["B"]["blocks_shifted_lin"] = _blocksparse_coords_v3(sidxpb_B, spm_B)
    case["D"]["blocks_shifted_lin"] = _blocksparse_coords_v3(sidxpb_D, spm_D)
    if case["C"] is not None:
        case["C"]["blocks_shifted_lin"] = _blocksparse_coords_v3(sidxpb_C, spm_C)

    return case


def reference_tensor_product_bs(A: sr.AbelianArray, B: sr.AbelianArray, C: Union[sr.AbelianArray, None],
                                a_modes, b_modes, c_modes, d_modes,
                                alpha, beta):
    assert (C is None) or tuple(c_modes) == tuple(d_modes), "expected modes of C to be the same as modes of D"
    # preprocess A, B, C, D to be in the format expected by tapp_torch.ops.tensor_product_bs
    # I. compute contraction to get structure of resulting tensor D
    matching_indices = [(i, j) for i, a in enumerate(a_modes) for j, b in enumerate(b_modes) if a == b]
    contract_idx_a, contract_idx_b= zip(*matching_indices) if len(matching_indices)>0 else ([],[])

    # II. compute permutation to get D in the order of d_modes
    uncontracted_idx_a, uncontracted_modes_a = zip(*[(i,a) for i, a in enumerate(a_modes) if a not in b_modes])
    uncontracted_idx_b, uncontracted_modes_b = zip(*[(j,b) for j, b in enumerate(b_modes) if b not in a_modes])
    # Find permutation needed to bring uncontracted_modes_a + uncontracted_modes_b into order of d_modes
    combined_modes = list(uncontracted_modes_a) + list(uncontracted_modes_b)
    assert set(combined_modes) == set(d_modes), "modes of D must be a permutation of the uncontracted modes of A and B"
    permutation_d = [combined_modes.index(mode) for mode in d_modes]
    
    D= alpha * sr.tensordot(A, B, axes=[contract_idx_a, contract_idx_b])
    D= D.transpose(permutation_d)
    if C is not None:
        D= D + beta * C
    return D


def reference_tensordot_bs(A: sr.AbelianArray, B: sr.AbelianArray,
                        a_contracted_modes, b_contracted_modes, out_modes=None):
    D= sr.tensordot(A, B, axes=[a_contracted_modes, b_contracted_modes])
    if out_modes is not None:
        D= D.transpose(tuple(out_modes))
    return D

    
class TestTensorProductBs(TestCase):
    def sample_inputs(self, dtype, device, *, requires_grad=False):
        """
        Returns a list of tuples (A, B, C, a_modes, b_modes, c_modes, d_modes, alpha, beta) where:
        
        result_{d_modes} = alpha A_{a_modes} x B_{b_modes}) + beta C_{c_modes}

        """
        dtype= str(dtype).split(".")[1] # e.g. torch.float32 -> "float32"
        samples= [ 
            (sr.utils.get_rand("Z2",
                ( sr.BlockIndex(chargemap={0:1,1:2}, dual=False),
                    sr.BlockIndex(chargemap={0:3,1:4}, dual=True) ),
                charge= 0, fermionic= False, dtype= dtype,),
            sr.utils.get_rand("Z2",
                ( sr.BlockIndex(chargemap={0:1,1:2}, dual=False),
                    sr.BlockIndex(chargemap={0:3,1:4}, dual=False) ),
                charge= 0, fermionic= False, dtype= dtype,),
            sr.utils.get_rand("Z2",
                ( sr.BlockIndex(chargemap={0:1,1:2}, dual=False),
                    sr.BlockIndex(chargemap={0:1,1:2}, dual=False) ),
                charge= 0, fermionic= False, dtype= dtype,),
            [0,1], [2,1], [0,2], [0,2], 1., 1.),

            (sr.utils.get_rand("Z3",
                ( sr.BlockIndex(chargemap={0:1,2:2}, dual=False),
                    sr.BlockIndex(chargemap={0:3,1:4,2:2}, dual=True),
                    sr.BlockIndex(chargemap={0:2,1:1,2:3}, dual=False) ),
                charge= 0, fermionic= False, dtype= dtype,),
            sr.utils.get_rand("Z3",
                ( sr.BlockIndex(chargemap={0:3,1:4,2:2}, dual=False),
                    sr.BlockIndex(chargemap={0:2,2:2}, dual=True),
                    sr.BlockIndex(chargemap={0:2,1:1,2:3}, dual=True) ),
                charge= 0, fermionic= False, dtype= dtype,),
            sr.utils.get_rand("Z3",
                ( sr.BlockIndex(chargemap={0:2,2:2}, dual=True),
                    sr.BlockIndex(chargemap={0:1,2:2}, dual=False) ),
                charge= 0, fermionic= False, dtype= dtype,),
            [0,1,2], [1,3,2], [3,0], [3,0], 1., 1.),
        ]
        return [ tuple( SymmrayAdapter.to_torch_(t,device=device) for t in sample[:3] ) + sample[3:] for sample in samples ]


    @parametrize("device", ["cuda"] if torch.cuda.is_available() else [])
    @parametrize("dtype", DTYPE_OPTIONS)
    def test_correctness(self, dtype, device):
        samples = self.sample_inputs(dtype,device)
        for sample in samples:
            expected= reference_tensor_product_bs(*sample)
            
            args_tapp_torch_bs= make_sample_tensor_product_bs( 
                *sample[:3], expected, *sample[3:], device=device, requires_grad=False)
            tapp_torch.ops.tensor_product_bs(*args_tapp_torch_bs)

            # fill in symmetric array
            R = SymmrayAdapter.fill(args_tapp_torch_bs[3], expected)

            torch.testing.assert_close(R.blocks, expected.blocks)

    @parametrize("device", ["cuda"] if torch.cuda.is_available() else [])
    @parametrize("dtype", DTYPE_OPTIONS)
    def test_opcheck(self, dtype, device):
        samples = self.sample_inputs(dtype, device)
        op = getattr(torch.ops, "tapp_torch").tensor_product_bs
        for args in samples:
            expected= reference_tensor_product_bs(*args)
            args_tapp_torch_bs= make_sample_tensor_product_bs(
                *args[:3], expected, *args[3:], device=device, requires_grad=False)
            opcheck(op, args_tapp_torch_bs)

            [ a.requires_grad_(True) for a in args_tapp_torch_bs[:3] ]
            opcheck(op, args_tapp_torch_bs)


class TestTensordotBs(TestCase):

    def sample_inputs(self, dtype, device, *, requires_grad=False):
        """
        Returns a list of tuples (A, B, a_contracted_modes, b_contracted_modes, out_modes) where:
        
        result_{out_modes} = alpha A_{a_other_modes}{a_contracted_modes} x B_{b_contracted_modes}{b_other_modes})

        """
        dtype= str(dtype).split(".")[1] # e.g. torch.float32 -> "float32"
        samples= [ 
            (sr.utils.get_rand("Z2",
                ( sr.BlockIndex(chargemap={0:1,1:2}, dual=False),
                    sr.BlockIndex(chargemap={0:3,1:4}, dual=True) ),
                charge= 0, fermionic= False, dtype= dtype,),
            sr.utils.get_rand("Z2",
                ( sr.BlockIndex(chargemap={0:1,1:2}, dual=False),
                    sr.BlockIndex(chargemap={0:3,1:4}, dual=False) ),
                charge= 0, fermionic= False, dtype= dtype,),
            [1], [1], None),

            (sr.utils.get_rand("Z3",
                ( sr.BlockIndex(chargemap={0:1,2:2}, dual=False),
                    sr.BlockIndex(chargemap={0:3,1:4,2:2}, dual=True),
                    sr.BlockIndex(chargemap={0:2,1:1,2:3}, dual=False) ),
                charge= 0, fermionic= False, dtype= dtype,),
            sr.utils.get_rand("Z3",
                ( sr.BlockIndex(chargemap={0:3,1:4,2:2}, dual=False),
                    sr.BlockIndex(chargemap={0:2,2:2}, dual=True),
                    sr.BlockIndex(chargemap={0:2,1:1,2:3}, dual=True) ),
                charge= 0, fermionic= False, dtype= dtype,),
            [1,2], [0,2], [1,0]),
        ]
        return [ tuple( SymmrayAdapter.to_torch_(t,device=device) for t in sample[:2] ) + sample[2:] for sample in samples ]

    @parametrize("device", ["cuda"] if torch.cuda.is_available() else [])
    @parametrize("dtype", DTYPE_OPTIONS)
    def test_correctness(self, dtype, device):
        samples = self.sample_inputs(dtype, device)
        for sample in samples:
            expected = reference_tensordot_bs(*sample)

            args_tapp_torch_bs= make_sample_tensordot_bs( 
                *sample[:2], expected, *sample[2:], device=device, requires_grad=False)
            D= tapp_torch.ops.tensordot_bs(*args_tapp_torch_bs)

            R = SymmrayAdapter.fill(D, expected)

            torch.testing.assert_close(R.blocks, expected.blocks)

    @parametrize("device", ["cuda"] if torch.cuda.is_available() else [])
    @parametrize("dtype", DTYPE_OPTIONS)
    @parametrize("test_utils", ["test_schema",
                                "test_autograd_registration",
                                "test_faketensor",
                                "test_aot_dispatch_dynamic"])
                                # "test_aot_dispatch_static", ])
    def test_opcheck_test(self, dtype, device, test_utils):
        samples = self.sample_inputs(dtype, device, requires_grad=False)
        op = getattr(torch.ops, "tapp_torch").tensordot_bs.default
        for args in samples:
            expected = reference_tensordot_bs(*args)
            args_tapp_torch_bs= make_sample_tensordot_bs( 
                *args[:2], expected, *args[2:], device=device, requires_grad=True)
            opcheck(op, args_tapp_torch_bs, test_utils=test_utils)
        
            [ a.requires_grad_(False) for a in args_tapp_torch_bs[:2] ]
            opcheck(op, args_tapp_torch_bs)

    @parametrize("device", ["cuda"] if torch.cuda.is_available() else [])
    @parametrize("dtype", DTYPE_OPTIONS)
    def test_gradients(self, dtype, device):
        samples = self.sample_inputs(dtype, device, requires_grad=False)
        for sample in samples:
            # get structure of the resulting block-sparse tensor D
            expected = reference_tensordot_bs(*sample)

            args_tapp_torch_bs= make_sample_tensordot_bs( 
                *sample[:2], expected, *sample[2:], device=device, requires_grad=True)
            
            diff_tensors = [a for a in args_tapp_torch_bs if isinstance(a, torch.Tensor) and a.requires_grad]
            out = tapp_torch.ops.tensordot_bs(*args_tapp_torch_bs)
            grad_out = torch.randn_like(out)
            grad_a, grad_b = torch.autograd.grad(out, diff_tensors, grad_out)

            def f_reference_tensordot_bs(a,b):
                # fill symmray arrays from torch tensors
                A = SymmrayAdapter.fill(a, sample[0])
                B = SymmrayAdapter.fill(b, sample[1])
                res = reference_tensordot_bs(A, B, *sample[2:])

                # flatten result back to torch tensor
                return SymmrayAdapter.flatten(res, device=device)[0]

            out = f_reference_tensordot_bs(*diff_tensors)
            expected_grad_a, expected_grad_b = torch.autograd.grad(out, diff_tensors, grad_out)

            G_a = SymmrayAdapter.fill(grad_a, sample[0])
            G_a_expected = SymmrayAdapter.fill(expected_grad_a, sample[0])
            torch.testing.assert_close(G_a.blocks, G_a_expected.blocks)

            G_b = SymmrayAdapter.fill(grad_b, sample[1])
            G_b_expected = SymmrayAdapter.fill(expected_grad_b, sample[1])
            torch.testing.assert_close(G_b.blocks, G_b_expected.blocks)


instantiate_parametrized_tests(TestTensorProductBs)
instantiate_parametrized_tests(TestTensordotBs)

if __name__ == "__main__":
    run_tests()