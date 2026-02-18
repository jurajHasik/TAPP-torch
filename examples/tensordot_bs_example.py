from typing import Optional, Sequence, Tuple
import torch
import tapp_torch
import argparse
from itertools import accumulate
import numpy as np
import symmray as sr


def parse_args():
    parser = argparse.ArgumentParser(description="Tensordot example with tapp_torch")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run on (default: cpu)")
    parser.add_argument("--dtype", type=str, default="float64", 
                        choices=["float32", "float64", "complex64", "complex128"],
                        help="Data type (default: float64)")
    return parser.parse_args()

def get_dtype(dtype_str):
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    return dtype_map[dtype_str]


class SymmrayAdapter:
    @staticmethod
    def to_torch_(T: sr.AbelianArray, device=None):
        """
        Convert the blocks of a Symmray AbelianArray, by default backed by NymPy, to torch.Tensor's in-place.
        """
        for b_idx, b in T.blocks.items():
            if isinstance(b, torch.Tensor):
                continue
            elif isinstance(b, np.ndarray):
                T.blocks[b_idx] = torch.from_numpy(b).to(device="cpu" if device is None else device)
            else:
                T.blocks[b_idx] = torch.tensor(b)
                if device is not None:
                    T.blocks[b_idx] = T.blocks[b_idx].to(device=device)
        return T

    @staticmethod
    def requires_grad_(T: sr.AbelianArray, requires_grad):
        """
        Set requires_grad for all blocks of a torch-backed symmray's AbelianArray in-place.
        """
        assert isinstance(next(iter(T.blocks.values())),torch.Tensor), "Expected blocks to be torch.Tensor's"
        for b_idx, b in T.blocks.items():
            if isinstance(b, torch.Tensor):
                b.requires_grad_(requires_grad)

    @staticmethod
    def reset_grad_(T: sr.AbelianArray):
        """
        Reset gradients for all blocks of a torch-backed symmray's AbelianArray in-place.
        """
        assert isinstance(next(iter(T.blocks.values())),torch.Tensor), "Expected blocks to be torch.Tensor's"
        for b_idx, b in T.blocks.items():
            if isinstance(b, torch.Tensor) and b.grad is not None:
                b.grad.zero_()

    @staticmethod
    def grad(T: sr.AbelianArray) -> sr.AbelianArray:
        res= T.copy()
        for b_idx, b in T.blocks.items():
            if isinstance(b, torch.Tensor):
                res.blocks[b_idx] = b.grad
            else:
                res.blocks[b_idx] = None
        return res

    @staticmethod
    def flatten(T: sr.AbelianArray, device=None):
        """
        Flatten blocks of symmray's AbelianArray into a 1D torch tensor.
        """
        blocks = list(T.blocks.values())
        sizes = [b.numel() for b in blocks] if isinstance(blocks[0], torch.Tensor) else [b.size for b in blocks]
        offsets = list(accumulate([0] + sizes[:-1]))
        slices = [
            (((offset, offset + size),), block.shape, size)
            for offset, size, block in zip(offsets, sizes, blocks)
        ]
        if device is not None:
            flat = torch.cat([b.reshape(-1) for b in blocks]).to(device=device)
        else:
            flat = torch.cat([b.reshape(-1) for b in blocks])
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

    # Build metadata for tensordot_bs
    @staticmethod
    def get_metadata(T: sr.AbelianArray, slices: Sequence[Tuple[ Tuple[int, int], Tuple[int, ...], int]]):
        num_modes = T.ndim
        blocks_list = list(T.blocks.keys())

        # Count sections per mode
        numSectionsPerMode = list([len(mode.sizes) for mode in T.indices])

        # Get extents
        sectionExtents = []
        sectionExtents = list([list(mode.sizes) for mode in T.indices])
        sectionExtents_linearized= sum(sectionExtents, [])

        # Block indices (linearized)
        blocks = sum([list(b_idx) for b_idx in blocks_list],[])

        # Strides
        strides = []
        for _, shape, _ in slices:
            stride = 1
            mode_strides = []
            for s in reversed(shape):
                mode_strides.insert(0, stride)
                stride *= s
            strides.extend(mode_strides)

        # Offsets
        offsets = [s[0][0][0] for s in slices]

        return numSectionsPerMode, sectionExtents_linearized, blocks, strides, offsets

    @staticmethod
    def tapp_tensordot_bs(A: sr.AbelianArray, B: sr.AbelianArray, 
                     contracted_modes_A: Sequence[int], contracted_modes_B: Sequence[int], 
                     modes_out: Optional[Sequence[int]]=None, device: Optional[str]=None):
        
        # Convert to torch
        assert isinstance(next(iter(A.blocks.values())),torch.Tensor), "Expected A blocks to be torch.Tensor's"
        assert isinstance(next(iter(B.blocks.values())),torch.Tensor), "Expected B blocks to be torch.Tensor's"

        # Flatten
        A_flat, A_slices = SymmrayAdapter.flatten(A)
        B_flat, B_slices = SymmrayAdapter.flatten(B)
        assert A_flat.device==B_flat.device, "Expected A and B blocks to reside on the same device"

        # NOTE: For simplicity, we are not implementing an isolated metadata computation, 
        #       i.e. resolving only the block-sparse structure of the result without performing the actual contraction.
        #       Here, for simplicity, we get it directly from symmray's own tensordot result
        D_expected = sr.tensordot(A, B, axes=[contracted_modes_A, contracted_modes_B])
        if modes_out is not None:
            D_expected = D_expected.transpose(modes_out)
        _, D_slices = SymmrayAdapter.flatten(D_expected, device=A_flat.device)

        D_flat= tapp_torch.ops.tensordot_bs(
            A_flat, B_flat, 
            contracted_modes_A, contracted_modes_B, 
            *SymmrayAdapter.get_metadata(A, A_slices), 
            *SymmrayAdapter.get_metadata(B, B_slices), 
            *SymmrayAdapter.get_metadata(D_expected, D_slices),
            modes_out=modes_out
        )

        # reconstruct the result as a Symmray AbelianArray
        D= SymmrayAdapter.fill(D_flat, D_expected)
        return D
    

def main():
    args = parse_args()
    device = args.device
    dtype = args.dtype

    A_np = sr.utils.get_rand(
        "Z3",
        (sr.BlockIndex(chargemap={0: 1, 1: 2, 2: 1}, dual=False),
         sr.BlockIndex(chargemap={0: 2, 1: 2, 2: 2}, dual=True),
         sr.BlockIndex(chargemap={0: 3, 1: 4, 2: 3}, dual=True)),
        charge=0, fermionic=False, dtype=dtype
    )
    B_np = sr.utils.get_rand(
        "Z3",
        (sr.BlockIndex(chargemap={0: 3, 1: 4, 2: 3}, dual=False),
         sr.BlockIndex(chargemap={0: 3, 1: 2, 2: 1}, dual=True),
         sr.BlockIndex(chargemap={0: 2, 1: 2, 2: 2}, dual=False)),
        charge=0, fermionic=False, dtype=dtype
    )
    C_np = sr.utils.get_rand(
        "Z3",
        (sr.BlockIndex(chargemap={0: 3, 1: 2, 2: 1}, dual=False),
         sr.BlockIndex(chargemap={0: 3, 1: 1, 2: 3}, dual=False)),
        charge=0, fermionic=False, dtype=dtype
    )

    A = SymmrayAdapter.to_torch_(A_np, device=device)
    B = SymmrayAdapter.to_torch_(B_np, device=device)
    C = SymmrayAdapter.to_torch_(C_np, device=device)
    SymmrayAdapter.requires_grad_(A,True)
    SymmrayAdapter.requires_grad_(B,True)
    SymmrayAdapter.requires_grad_(C,True)

    # Default symmray's tensordot
    # Contract A modes [1,2] with B modes [0,1]
    tmp = sr.tensordot(A, B, [[1, 2], [2, 0]])
    out = sr.tensordot(tmp, C, [[1], [0]])
    out = out.transpose([1,0])
    loss = out.sum().norm()
    loss.backward()
    
    print("symmray tensordot default")
    print("min(grad A), max(grad A):", SymmrayAdapter.grad(A).min().item(), SymmrayAdapter.grad(A).max().item())
    print("min(grad B), max(grad B):", SymmrayAdapter.grad(B).min().item(), SymmrayAdapter.grad(B).max().item())
    print("min(grad C), max(grad C):", SymmrayAdapter.grad(C).min().item(), SymmrayAdapter.grad(C).max().item())

    # Contract A modes [1,2] with B modes [0,1]
    SymmrayAdapter.reset_grad_(A)
    SymmrayAdapter.reset_grad_(B)
    SymmrayAdapter.reset_grad_(C)
    tmp = SymmrayAdapter.tapp_tensordot_bs(A, B, [1, 2], [2, 0])
    out = SymmrayAdapter.tapp_tensordot_bs(tmp, C, [1], [0], modes_out=[1,0])
    loss1 = out.sum().norm()
    loss1.backward()

    print("symmray tapp_torch.tensordot_bs")
    print("loss close:", torch.allclose(loss, loss1, atol=1e-6, rtol=1e-6))
    print("min(grad A), max(grad A):", SymmrayAdapter.grad(A).min().item(), SymmrayAdapter.grad(A).max().item())
    print("min(grad B), max(grad B):", SymmrayAdapter.grad(B).min().item(), SymmrayAdapter.grad(B).max().item())
    print("min(grad C), max(grad C):", SymmrayAdapter.grad(C).min().item(), SymmrayAdapter.grad(C).max().item())

    # torch.compile path
    # NOTE: Needs dedidicated metatdata computation in SymmrayAdapter.tapp_tensordot_bs 
    # def f(A, B, C):
    #     tmp = SymmrayAdapter.tapp_tensordot_bs(A, B, [1, 2], [2, 0])
    #     out = SymmrayAdapter.tapp_tensordot_bs(tmp, C, [1], [0], modes_out=[1,0])
    #     loss = out.sum().norm()
    #     return loss
    
    # A.grad, B.grad, C.grad = None, None, None  # reset grads
    # SymmrayAdapter.reset_grad_(A)
    # SymmrayAdapter.reset_grad_(B)
    # SymmrayAdapter.reset_grad_(C)
    # compiled = torch.compile(f)
    # loss2 = compiled(A, B, C)
    # loss2.backward()


if __name__ == "__main__":
    main()