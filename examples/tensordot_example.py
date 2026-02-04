import torch
import tapp_torch


def main():
    device = "cpu"
    dtype = torch.float64

    A = torch.randn(3, 4, 5, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(5, 4, 2, device=device, dtype=dtype, requires_grad=True)
    C = torch.randn(3, 2, device=device, dtype=dtype, requires_grad=True)

    # Contract A modes [2,1] with B modes [0,1]
    tmp = tapp_torch.ops.tensordot(A, B, [2, 1], [0, 1])
    out = tapp_torch.ops.tensordot(tmp, C, [1], [1], modes_out=[1,0])
    loss = out.pow(2).sum()
    loss.backward()

    print("out shape:", out.shape)
    print("grad A:", A.grad.abs().mean().item())
    print("grad B:", B.grad.abs().mean().item())
    print("grad C:", C.grad.abs().mean().item())

    # torch.compile path
    def f(A, B, C):
        tmp= tapp_torch.ops.tensordot(A, B, [2, 1], [0, 1])
        out= tapp_torch.ops.tensordot(tmp, C, [1], [1], modes_out=[1,0])
        return out
    
    A.grad, B.grad, C.grad = None, None, None  # reset grads
    compiled = torch.compile(f)
    out2 = compiled(A, B, C)
    loss2 = out2.pow(2).sum()
    loss2.backward()

    print("compiled out close:", torch.allclose(out, out2, atol=1e-6, rtol=1e-6))
    print("grad A:", A.grad.abs().mean().item())
    print("grad B:", B.grad.abs().mean().item())
    print("grad C:", C.grad.abs().mean().item())

if __name__ == "__main__":
    main()