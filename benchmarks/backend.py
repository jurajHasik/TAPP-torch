"""
Custom autoray backend that dispatches tensordot and permute operations
to tapp_torch, while falling back to PyTorch for everything else.
"""
import autoray
import torch
import tapp_torch
import os

BACKEND_TORCH = "torch_default"
BACKEND_TAPP = "tapp_torch"
TAPP_LOG_LEVEL = int(os.environ.get('TAPP_LOG_LEVEL', '0'))

if TAPP_LOG_LEVEL > 5:
    def _tensordot_default(a,b, *args, **kwargs):
        """Fallback tensordot that dispatches to torch.tensordot."""
        torch.cuda.nvtx.range_push(f"torch.tensordot_{list(a.shape)}x{list(b.shape)}")
        res= torch.tensordot(a,b, *args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return res
else:
    def _tensordot_default(a,b, *args, **kwargs):
        """Fallback tensordot that dispatches to torch.tensordot."""
        return torch.tensordot(a,b, *args, **kwargs)

def _tensordot_tapp(a, b, axes=2):
    """
    autoray-compatible tensordot that dispatches to tapp_torch.ops.tensordot.

    Parameters
    ----------
    a : torch.Tensor
    b : torch.Tensor
    axes : int or (list[int], list[int])
        If an int N, contract the last N modes of a with the first N of b.
        If a tuple/list of two lists, explicit mode indices.
    """
    if isinstance(axes, int):
        contracted_modes_A = list(range(a.dim() - axes, a.dim()))
        contracted_modes_B = list(range(axes))
    else:
        contracted_modes_A, contracted_modes_B = axes
        contracted_modes_A = list(contracted_modes_A)
        contracted_modes_B = list(contracted_modes_B)

    return tapp_torch.ops.tensordot(a, b, contracted_modes_A, contracted_modes_B)


def _transpose(a, axes=None):
    """
    autoray-compatible transpose/permute that could leverage tapp_torch
    in the future. For now wraps torch.permute (which is zero-copy).

    Parameters
    ----------
    a : torch.Tensor
    axes : tuple[int, ...] or None
        Permutation of dimensions. ``None`` reverses them.
    """
    if axes is None:
        axes = tuple(range(a.dim() - 1, -1, -1))
    return a.permute(axes)


def setup_backends():
    """Register the tapp_torch backend with autoray."""
    # Register tensordot
    autoray.register_function(
        BACKEND_TAPP,
        "tensordot",
        _tensordot_tapp,
    )

    # Register transpose (permute)
    autoray.register_function(
        BACKEND_TAPP,
        "transpose",
        _transpose,
    )

    autoray.register_function(
        BACKEND_TORCH,
        "tensordot",
        _tensordot_default,
    )