import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from torch.testing._internal.optests import opcheck


def reference_tensor_product(a, b, c, d,
                             modes_a, modes_b, modes_c, modes_d,
                             alpha, beta):
    d= alpha * torch.einsum(a, modes_a, b, modes_b, modes_d) + beta * c 
    return d

def reference_tensordot(a, b, contract_idx_a, contract_idx_b, out_order= None):
    res= torch.tensordot(a, b, dims=(contract_idx_a, contract_idx_b))
    res= res.permute(out_order) if out_order is not None else res
    return res

import tapp_torch


class TestTensorProduct(TestCase):
    def sample_inputs(self, dtype, device, *, requires_grad=False):
        def make_tensor(*size):
            if len(size)==0:
                return torch.tensor(1., dtype=dtype, device=device, requires_grad=requires_grad)
            return torch.randn(size, dtype=dtype, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            if len(size)==0:
                return torch.tensor(1., dtype=dtype, device=device, requires_grad=False)
            return torch.randn(size, dtype=dtype, device=device, requires_grad=False)

        return [
            [make_tensor(1), make_tensor(1), make_tensor(1), make_tensor(1), [1,], [1,], [1,], [1,], 1, 1],
            # [make_tensor(3), make_tensor(3), None, make_tensor(), [1,], [1,], None, [], 1, 1],
            # [make_tensor(3,3), make_tensor(3,3), None, make_tensor(3,3), [0,1], [1,2], None, [0,2], 1, 1],
        ]

    @parametrize("device", ["cpu",])
    @parametrize("dtype", [torch.float64,])
    def test_correctness(self, dtype, device):
        samples = self.sample_inputs(dtype,device)
        for args in samples:
            tapp_torch.ops.tensor_product(*args)
            expected = reference_tensor_product(*args)
            torch.testing.assert_close(args[3], expected)
    

    # @parametrize("device", ["cpu",])
    # def test_gradients(self, device):
    #     samples = self.sample_inputs(device, requires_grad=True)
    #     for args in samples:
    #         diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
    #         out = tapp_torch.ops.tensor_product(*args)
    #         grad_out = torch.randn_like(out)
    #         result = torch.autograd.grad(out, diff_tensors, grad_out)

    #         out = reference_tensor_product(*args)
    #         expected = torch.autograd.grad(out, diff_tensors, grad_out)

    #         torch.testing.assert_close(result, expected)

    @parametrize("device", ["cpu",])
    @parametrize("dtype", [torch.float64,])
    def test_opcheck(self, dtype, device):
        samples = self.sample_inputs(dtype, device, requires_grad=True)
        samples.extend(self.sample_inputs(dtype, device, requires_grad=False))
        op = getattr(torch.ops, "tapp_torch").tensor_product.default
        for args in samples:
            opcheck(op, args)


class TestTensordot(TestCase):

    def sample_inputs(self, dtype, device, *, requires_grad=False):
        def make_tensor(*size):
            if len(size)==0:
                return torch.tensor(1., dtype=dtype, device=device, requires_grad=requires_grad)
            return torch.randn(size, dtype=dtype, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            if len(size)==0:
                return torch.tensor(1., dtype=dtype, device=device, requires_grad=False)
            return torch.randn(size, dtype=dtype, device=device, requires_grad=False)

        return [
            [make_tensor(1,1), make_tensor(1,1).conj(), [1,], [0,]],
            [make_tensor(1).conj(), make_tensor(1), [0,], [0,]],
            [make_tensor(2), make_tensor(2).conj(), [], []],
            [make_tensor(2,3), make_tensor(2,3), [0,], [0,]],
            [make_tensor(2,3), make_tensor(2,3).conj(), [1,], [1,]],
            [make_tensor(2,3,2), make_tensor(2,2,3), [1,2], [2,1]],
            [make_tensor(2,3,4,5).conj(), make_tensor(2,3,5), [0,3], [0,2]],
            [make_tensor(2,3,5), make_tensor(2,3,4,5), [2,0], [3,0]],
            [make_tensor(2,3), make_tensor(2,3), [0,], [0,], [1,0]],
            [make_tensor(2,3).conj(), make_tensor(2,3).conj(), [1,], [1,], [1,0]],
            [make_tensor(2,3,2).conj(), make_tensor(2,2,3).conj(), [1,2], [2,1], [1,0]],
            [make_tensor(2,3,2), make_tensor(2,2,3), [1,2], [2,1], [1,0]],
            [make_tensor(2,3,4,5).conj(), make_tensor(2,3,5), [0,3], [0,2], [1,2,0]],
            [make_tensor(2,3,5).conj(), make_tensor(2,3,4,5).conj(), [2,0], [3,0], [2,1,0]],
        ]

    @parametrize("device", ["cpu",])
    @parametrize("dtype", [torch.float64,torch.complex128])
    def test_correctness(self, dtype, device):
        samples = self.sample_inputs(dtype,device)
        for args in samples:
            result= tapp_torch.ops.tensordot(*args)
            expected = reference_tensordot(*args)
            torch.testing.assert_close(result, expected)

    @parametrize("device", ["cpu",])
    @parametrize("dtype", [torch.float64,torch.complex128])
    @parametrize("test_utils", ["test_schema",
                                "test_autograd_registration",
                                "test_faketensor",
                                "test_aot_dispatch_static", 
                                "test_aot_dispatch_dynamic"])
    def test_opcheck_test_schema(self, dtype, device, test_utils):
        samples = self.sample_inputs(dtype, device, requires_grad=True)
        samples.extend(self.sample_inputs(dtype, device, requires_grad=False))
        op = getattr(torch.ops, "tapp_torch").tensordot.default
        for args in samples:
            opcheck(op, args, test_utils=test_utils)

    @parametrize("device", ["cpu",])
    @parametrize("dtype", [torch.float64,torch.complex128])
    def test_gradients(self, dtype, device):
        samples = self.sample_inputs(dtype, device, requires_grad=True)
        for args in samples:
            diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
            out = tapp_torch.ops.tensordot(*args)
            grad_out = torch.randn_like(out)
            result = torch.autograd.grad(out, diff_tensors, grad_out)

            out = reference_tensordot(*args)
            expected = torch.autograd.grad(out, diff_tensors, grad_out)

            torch.testing.assert_close(result, expected)


instantiate_parametrized_tests(TestTensorProduct)
instantiate_parametrized_tests(TestTensordot)

if __name__ == "__main__":
    run_tests()