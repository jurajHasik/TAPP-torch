from itertools import accumulate
import operator
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
import os

def make_test_case(filename, dtype, device, requires_grad=False):
    with open(filename, "r") as f:
        case = json.load(f)
    
    a_modes= case["modes_A"]
    b_modes= case["modes_B"]
    d_modes= case["modes_D"]
    c_modes= None if case["C"] is None else case["modes_C"]

    alpha= case["alpha"]
    beta= case["beta"]

    A= torch.rand(case["A"]["struct"]["size"], dtype=dtype, device=device, requires_grad=requires_grad)
    B= torch.rand(case["B"]["struct"]["size"], dtype=dtype, device=device, requires_grad=requires_grad)
    D= torch.zeros(case["D"]["struct"]["size"], dtype=dtype, device=device)
    C= None if case["C"] is None else torch.rand(case["C"]["struct"]["size"], dtype=dtype, device=device, requires_grad=requires_grad)

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
        section_idx_per_block= np.array(T["struct"]["t"])
        section_extents_per_block= np.array(T["struct"]["D"])
        sections_per_mode= np.asarray([list(sorted(set(section_idx_per_block[:, i]))) for i in range(section_idx_per_block.shape[1])])
        sections_per_mode= sections_per_mode if sections_per_mode.ndim==3 else sections_per_mode[..., np.newaxis]
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

    # II. linearize section extents
    a_numSectionsPerMode= [len(s) for s in spm_A]
    b_numSectionsPerMode= [len(s) for s in spm_B]
    d_numSectionsPerMode= [len(s) for s in spm_D]
    c_numSectionsPerMode= None if case["C"] is None else [len(s) for s in spm_C] 
    a_sectionExtents= sum([extents for extents in epm_A],[])
    b_sectionExtents= sum([extents for extents in epm_B],[])
    d_sectionExtents= sum([extents for extents in epm_D],[])
    c_sectionExtents= None if case["C"] is None else sum([extents for extents in epm_C],[])

    def compute_offsets_and_strides(T):
        slices= T["slices"]
        S= np.empty( (len(slices),len(slices[0][1])), dtype=np.int64 )
        S[:,-1]=1
        Ds= np.asarray( tuple(b[1] for b in slices) )
        np.cumprod( Ds[:, -1:0:-1], axis=-1, out= S[:,:len(slices[0][1])-1][:,::-1])
        return tuple(s[0][0][0] for s in slices), S.reshape(-1).tolist()

    a_offsets, a_strides= compute_offsets_and_strides(case["A"])
    b_offsets, b_strides= compute_offsets_and_strides(case["B"])
    d_offsets, d_strides= compute_offsets_and_strides(case["D"])
    c_offsets, c_strides= (None, None) if case["C"] is None else compute_offsets_and_strides(case["C"])
    
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

    a_blocks = _blocksparse_coords_v3(sidxpb_A, spm_A)
    b_blocks = _blocksparse_coords_v3(sidxpb_B, spm_B)
    d_blocks = _blocksparse_coords_v3(sidxpb_D, spm_D)
    c_blocks = None if case["C"] is None else _blocksparse_coords_v3(sidxpb_C, epm_C)

    return A, B, C, D, \
        a_modes, a_numSectionsPerMode, a_sectionExtents, a_blocks, a_strides, a_offsets, \
        b_modes, b_numSectionsPerMode, b_sectionExtents, b_blocks, b_strides, b_offsets, \
        c_modes, c_numSectionsPerMode, c_sectionExtents, c_blocks, c_strides, c_offsets, \
        d_modes, d_numSectionsPerMode, d_sectionExtents, d_blocks, d_strides, d_offsets, \
        alpha, beta


class TestTensorProductBs(TestCase):
    def sample_inputs(self, dtype, device, *, requires_grad=False):
        test_dir = os.path.dirname(__file__)
        return [
            [
                *make_test_case(os.path.join(test_dir, "inputs", "case1.json"), dtype, device, requires_grad)
            ]
        ]

    @parametrize("device", ["cuda"] if torch.cuda.is_available() else [])
    @parametrize("dtype", DTYPE_OPTIONS)
    def test_correctness(self, dtype, device):
        samples = self.sample_inputs(dtype,device)
        for args in samples:
            tapp_torch.ops.tensor_product_bs(*args)
            # expected = reference_tensor_product(*args)
            # torch.testing.assert_close(args[3], expected)

    # @parametrize("device", ["cuda"] if torch.cuda.is_available() else [])
    # @parametrize("dtype", DTYPE_OPTIONS)
    # def test_opcheck(self, dtype, device):
    #     samples = self.sample_inputs(dtype, device, requires_grad=True)
    #     samples.extend(self.sample_inputs(dtype, device, requires_grad=False))
    #     op = getattr(torch.ops, "tapp_torch").tensor_product_bs
    #     for args in samples:
    #         opcheck(op, args)


# class TestTensordot(TestCase):

#     def sample_inputs(self, dtype, device, *, requires_grad=False):
#         def make_tensor(*size):
#             if len(size)==0:
#                 return torch.tensor(1., dtype=dtype, device=device, requires_grad=requires_grad)
#             return torch.randn(size, dtype=dtype, device=device, requires_grad=requires_grad)

#         def make_nondiff_tensor(*size):
#             if len(size)==0:
#                 return torch.tensor(1., dtype=dtype, device=device, requires_grad=False)
#             return torch.randn(size, dtype=dtype, device=device, requires_grad=False)

#         return [
#             [make_tensor(1,1), make_tensor(1,1).conj(), [1,], [0,]],
#             [make_tensor(1).conj(), make_tensor(1), [0,], [0,]],
#             [make_tensor(2), make_tensor(2).conj(), [], []],
#             [make_tensor(2,3), make_tensor(2,3), [0,], [0,]],
#             [make_tensor(2,3), make_tensor(2,3).conj(), [1,], [1,]],
#             [make_tensor(2,3,2), make_tensor(2,2,3), [1,2], [2,1]],
#             [make_tensor(2,3,4,5).conj(), make_tensor(2,3,5), [0,3], [0,2]],
#             [make_tensor(2,3,5), make_tensor(2,3,4,5), [2,0], [3,0]],
#             [make_tensor(2,3), make_tensor(2,3), [0,], [0,], [1,0]],
#             [make_tensor(2,3).conj(), make_tensor(2,3).conj(), [1,], [1,], [1,0]],
#             [make_tensor(2,3,2).conj(), make_tensor(2,2,3).conj(), [1,2], [2,1], [1,0]],
#             [make_tensor(2,3,2), make_tensor(2,2,3), [1,2], [2,1], [1,0]],
#             [make_tensor(2,3,4,5).conj(), make_tensor(2,3,5), [0,3], [0,2], [1,2,0]],
#             [make_tensor(2,3,5).conj(), make_tensor(2,3,4,5).conj(), [2,0], [3,0], [2,1,0]],
#         ]

#     @parametrize("device", ["cpu","cuda"] if torch.cuda.is_available() else ["cpu",])
#     @parametrize("dtype", DTYPE_OPTIONS)
#     def test_correctness(self, dtype, device):
#         samples = self.sample_inputs(dtype,device)
#         for args in samples:
#             result= tapp_torch.ops.tensordot(*args)
#             expected = reference_tensordot(*args)
#             torch.testing.assert_close(result, expected)

#     @parametrize("device", ["cpu","cuda"] if torch.cuda.is_available() else ["cpu",])
#     @parametrize("dtype", DTYPE_OPTIONS)
#     @parametrize("test_utils", ["test_schema",
#                                 "test_autograd_registration",
#                                 "test_faketensor",
#                                 "test_aot_dispatch_static", 
#                                 "test_aot_dispatch_dynamic"])
#     def test_opcheck_test(self, dtype, device, test_utils):
#         samples = self.sample_inputs(dtype, device, requires_grad=True)
#         samples.extend(self.sample_inputs(dtype, device, requires_grad=False))
#         op = getattr(torch.ops, "tapp_torch").tensordot.default
#         for args in samples:
#             opcheck(op, args, test_utils=test_utils)

#     @parametrize("device", ["cpu","cuda"] if torch.cuda.is_available() else ["cpu",])
#     @parametrize("dtype", DTYPE_OPTIONS)
#     def test_gradients(self, dtype, device):
#         samples = self.sample_inputs(dtype, device, requires_grad=True)
#         for args in samples:
#             diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
#             out = tapp_torch.ops.tensordot(*args)
#             grad_out = torch.randn_like(out)
#             result = torch.autograd.grad(out, diff_tensors, grad_out)

#             out = reference_tensordot(*args)
#             expected = torch.autograd.grad(out, diff_tensors, grad_out)

#             torch.testing.assert_close(result, expected)


instantiate_parametrized_tests(TestTensorProductBs)
# instantiate_parametrized_tests(TestTensordot)

if __name__ == "__main__":
    run_tests()