from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor

__all__ = ["tensor_product","tensordot"]


def tensor_product(A: Tensor, B: Tensor, C: Union[Tensor,None], D: Tensor, 
                   modes_A: Sequence[int], modes_B: Sequence[int], modes_C: Union[Sequence[int],None], modes_D: Sequence[int],
                   alpha: Union[float,complex,Tensor,None], beta: Union[float,complex,Tensor,None]) -> None:
    """Performs D <- a*A*B+b*C. in an efficient fused kernel"""
    # preprocess to comply with Pytorch op schema https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func
    #
    # Under (previous) non-stable ABI Python's float and complex are mapped to C++ double and std::complex<double>
    alpha_t, beta_t= None, None
    if isinstance(alpha, Tensor):
        alpha_t= alpha
    elif isinstance(alpha, float) and not D.is_complex():
        alpha_t= torch.tensor(alpha, dtype= torch.float64, device='cpu')
    elif (isinstance(alpha, float) and D.is_complex()) or isinstance(alpha, complex):
        alpha_t= torch.tensor(alpha, dtype= torch.complex128, device='cpu')
    if isinstance(beta, Tensor):
        beta_t= beta
    elif isinstance(beta, float) and not D.is_complex():
        beta_t= torch.tensor(beta, dtype= torch.float64, device='cpu')
    elif (isinstance(beta, float) and D.is_complex()) or isinstance(beta, complex):
        beta_t= torch.tensor(beta, dtype= torch.complex128, device='cpu')
    return torch.ops.tapp_torch.tensor_product.default(A,B,C,D,
        modes_A,modes_B,modes_C,modes_D,alpha_t,beta_t)


# NOTE signature for contracted_modes is not supported by torch custom_op 
#
# def tensordot(A: Tensor, B: Tensor, 
#               contracted_modes: Union[int,Tuple[List[int], List[int]],List[List[int]],Tensor]) -> Tensor:
# contracted_modes: Specifies the modes to contract over. It can be:
# - An integer N: contracts the last N modes of A with the first N modes of B.
# - A tuple of two lists: the first list contains the modes of A to contract,
#   and the second list contains the modes of B to contract (indexed by position).
# - A list of two lists: similar to the tuple case.
# - Tensor: A 2D integer-valued tensor where the first row contains modes of A and
#        the second row contains modes of B to contract.
#
@torch.library.custom_op("tapp_torch::tensordot", mutates_args=())
def tensordot(A: Tensor, B: Tensor, contracted_modes_A: List[int], contracted_modes_B: List[int],
              modes_out: Optional[List[int]]=None) -> Tensor:
    """

    Performs standard tensordot contraction between A and B over the specified modes.
    [See https://docs.pytorch.org/docs/stable/generated/torch.tensordot.html]

    Parameters:
    A (Tensor): The first input tensor.
    B (Tensor): The second input tensor.
    contracted_modes_A (List[int]): List of modes in A to contract (indexed by position).
    contracted_modes_B (List[int]): List of modes in B to contract (...).
    modes_out (List[int], optional): Desired order of output modes. If None, the default order, corresponding
        to modes_out=[0, 1, 2, ..., N-1] where N is the number of remaining modes after contraction, is used.

    Returns:
    Tensor: The resulting tensor after performing the tensordot operation.
            The order of the remaining modes is preserved from A followed by B, unless modes_out is specified.
    """
    # Determine the output shape and reindex modes to match the tensor_product api
    modes_A= list(range(A.dim()))
    modes_B= [modes_A[contracted_modes_A[contracted_modes_B.index(n)]] if n in contracted_modes_B else j 
              for n,j in enumerate(range(A.dim(), A.dim()+B.dim()))]
    remaining_modes_A = [i for i in modes_A if i not in contracted_modes_A]
    remaining_modes_B = [j for n,j in enumerate(modes_B) if n not in contracted_modes_B]
    output_shape = [A.size(i) for i in remaining_modes_A] + [B.size(n) for n,_ in enumerate(modes_B) if n not in contracted_modes_B]
    modes_D= remaining_modes_A + remaining_modes_B
    if modes_out is not None:
        assert len(modes_out)==len(modes_D), "modes_out must have the same length as the number of remaining modes"
        output_shape = [output_shape[i] for i in modes_out]
        modes_D = [modes_D[i] for i in modes_out]

    # Create an output tensor with the computed shape
    D = torch.empty(output_shape, dtype=A.dtype, device=A.device)
    
    # Perform the tensor product with alpha=1 and beta=0
    tensor_product(A, B, None, D, modes_A, modes_B, None, modes_D,
                   alpha=1.0, beta=0.0)
    return D



# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.

# Use register_fake to add a ``FakeTensor`` kernel for the operator
@tensordot.register_fake
def _(A, B, contracted_modes_A, contracted_modes_B, modes_out=None):
    # Determine the output shape and reindex modes to match the tensor_product api
    modes_A= list(range(A.dim()))
    modes_B= [modes_A[contracted_modes_A[contracted_modes_B.index(n)]] if n in contracted_modes_B else j 
              for n,j in enumerate(range(A.dim(), A.dim()+B.dim()))]
    remaining_modes_A = [i for i in modes_A if i not in contracted_modes_A]
    # remaining_modes_B = [j for n,j in enumerate(modes_B) if n not in contracted_modes_B]
    output_shape = [A.size(i) for i in remaining_modes_A] + [B.size(n) for n,_ in enumerate(modes_B) if n not in contracted_modes_B]
    if modes_out is not None:
        output_shape = [output_shape[i] for i in modes_out]

    # Create an output tensor with the computed shape
    D = torch.empty(output_shape, dtype=A.dtype, device=A.device)
    return D

def _backward_tensordot(ctx, grad_D):
    """
    A_a,in B_in,b = D_ab => 
        dA_a,in = dD_ab B_b,in 
        dB_in,b = A_in,a dD_ab 

    dA, dB= dD . B^T, A^T . dD
    """
    A, B = ctx.saved_tensors
    shape_A, shape_B= ctx.shape_A, ctx.shape_B
    cidx_fwd_a= ctx.contracted_modes_A
    cidx_fwd_b= ctx.contracted_modes_B
    out_modes= ctx.out_modes
    grad_A, grad_B = None, None

    if out_modes is None:
        out_modes= list(range(grad_D.dim())) # default order

    invert_perm = lambda perm: [perm.index(i) for i in range(len(perm))]
    oidx_fwd_a= [i for i in range(len(shape_A)) if i not in cidx_fwd_a] # outgoing modes of A in forward
    idx_fwd_a= oidx_fwd_a + cidx_fwd_a                                  # permutation of A in forward
    oidx_fwd_b= [i for i in range(len(shape_B)) if i not in cidx_fwd_b] # outgoing modes of B in forward
    idx_fwd_b= cidx_fwd_b + oidx_fwd_b                                  # permutation of B in forward

    # forward(formally): 
    #   A_in->A_a,c and B_in->B_c,b => A_a,c B_c,b = D_ab => D_ab -> D_out
    if ctx.needs_input_grad[0]: 
        # dA_a,in = dD_ab B_b,in
        #
        cidx_bwd_b= oidx_fwd_b                                                       # contracted modes of B in backward
        cidx_bwd_d= [out_modes.index(grad_D.dim()-len(oidx_fwd_b)+n) for n,_ in enumerate(oidx_fwd_b)] # contracted modes of D in backward

        # grad_A shape is identical to shape of A 
        #   <=> to (outgoing indices of A <=> indices of grad_D without outgoing indices of B) & ingoing indices of B
        # TODO infer without relying on saved shapes

        # Assign labels to modes
        modes_B= list(range(B.dim()))
        modes_gD= list(range(B.dim(),B.dim()+grad_D.dim()))
        for n,i in enumerate(cidx_bwd_d): modes_gD[i]= modes_B[cidx_bwd_b[n]]

        modes_gA= [None]*len(shape_A)
        for n,i in enumerate(oidx_fwd_a): modes_gA[i]= modes_gD[out_modes.index(n)]
        for n,i in enumerate(cidx_fwd_a): modes_gA[i]= modes_B[cidx_fwd_b[n]]

        grad_A= torch.empty_like(A) if A is not None else torch.empty(shape_A, dtype=grad_D.dtype, device=grad_D.device)
        tensor_product(grad_D, B.conj(), None, grad_A, modes_gD, modes_B, None, modes_gA,
                   alpha=1.0, beta=0.0)

    if ctx.needs_input_grad[1]:
        # dB_in,b = A_in,a dD_ab 
        # 
        cidx_bwd_a= oidx_fwd_a                                      # contracted modes of A in backward
        cidx_bwd_d= [out_modes.index(n) for n,_ in enumerate(oidx_fwd_a)] # contracted modes of D in backward

        # Assign labels to modes
        modes_A= list(range(A.dim()))
        modes_gD= list(range(A.dim(),A.dim()+grad_D.dim()))
        for n,i in enumerate(cidx_bwd_d): modes_gD[i]= modes_A[cidx_bwd_a[n]]

        modes_gB= [None]*len(shape_B)
        for n,i in enumerate(oidx_fwd_b): modes_gB[i]= modes_gD[out_modes.index(grad_D.dim()-len(oidx_fwd_b)+n)]
        for n,i in enumerate(cidx_fwd_b): modes_gB[i]= modes_A[cidx_fwd_a[n]]

        grad_B= torch.empty_like(B) if B is not None else torch.empty(shape_B, dtype=grad_D.dtype, device=grad_D.device)
        tensor_product(A.conj(), grad_D, None, grad_B, modes_A, modes_gD, None, modes_gB,
                   alpha=1.0, beta=0.0)

    return grad_A, grad_B, None, None, None

def _setup_context_tensordot(ctx, inputs, output):
    A, B, contracted_modes_A, contracted_modes_B, out_modes= inputs
    ctx.shape_A, ctx.shape_B= A.shape, B.shape
    ctx.contracted_modes_A= contracted_modes_A
    ctx.contracted_modes_B= contracted_modes_B
    ctx.out_modes= out_modes
    
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = B
    if ctx.needs_input_grad[1]:
        saved_a = A
    ctx.save_for_backward(saved_a, saved_b)


# # This adds training support for the operator. You must provide us
# # the backward formula for the operator and a `setup_context` function
# # to save values to be used in the backward.
torch.library.register_autograd(
    "tapp_torch::tensordot", _backward_tensordot, setup_context=_setup_context_tensordot)


