#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cutensor.h>
#include <cuda_runtime.h>
#include <cutensor_bind.h>

// Force debug info for tensor_info struct
static volatile struct tensor_info* _dbg_tensor_info_dummy = nullptr;

#include <complex>
// #include <stdexcept>
#include <vector>
#include <string>

namespace tapp_torch {

namespace {

// Convert PyTorch dtype to TAPP datatype
TAPP_datatype get_tapp_dtype(const torch::stable::Tensor& tensor) {
    auto dtype = tensor.scalar_type();
    switch (dtype) {
        case torch::headeronly::ScalarType::Float:
            return TAPP_F32;
        case torch::headeronly::ScalarType::Double:
            return TAPP_F64;
        case torch::headeronly::ScalarType::ComplexFloat:
            return TAPP_C32;
        case torch::headeronly::ScalarType::ComplexDouble:
            return TAPP_C64;
        default:
            throw std::runtime_error("Unsupported (torch) tensor dtype for TAPP contraction");
    }
}

std::vector<int64_t> get_extents(const torch::stable::Tensor& tensor) {
  auto sizes = tensor.sizes();
  return std::vector<int64_t>(sizes.begin(), sizes.end());
}

std::vector<int64_t> get_strides(const torch::stable::Tensor& tensor) {
  auto strides = tensor.strides();
  return std::vector<int64_t>(strides.begin(), strides.end());
}

}


// D <- a*A*B+b*C.
template <typename scalar_t>
void tensor_product_impl_cuda(
  const torch::stable::Tensor& A,
  const torch::stable::Tensor& B,
  const torch::stable::Tensor& C,
  torch::stable::Tensor& D,
  const std::vector<int64_t>& idx_A,
  const std::vector<int64_t>& idx_B,
  const std::optional<std::vector<int64_t>>& idx_C,
  const std::vector<int64_t>& idx_D,
  scalar_t alpha,
  scalar_t beta) {

  STD_TORCH_CHECK(A.scalar_type() == B.scalar_type() && A.scalar_type() == D.scalar_type(), 
  "All tensors must have the same dtype");
  STD_TORCH_CHECK(A.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(B.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(D.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(static_cast<int64_t>(idx_A.size()) == A.dim(), 
      "Index vector length must match tensor A dimensions");
  STD_TORCH_CHECK(static_cast<int64_t>(idx_B.size()) == B.dim(), 
      "Index vector length must match tensor B dimensions");
  STD_TORCH_CHECK(static_cast<int64_t>(idx_D.size()) == D.dim(), 
      "Index vector length must match tensor D dimensions");
  STD_TORCH_CHECK((C.defined() && idx_C.has_value()) || (!C.defined() && !idx_C.has_value()),
    "Either both C tensor and idx_C must be provided, or neither");
  STD_TORCH_CHECK(C.defined() || beta == scalar_t(0), "If beta is non-zero, tensor C must be provided");

  if (C.defined()) {
    // TODO if c is provided, verify it matches output shape - d  
    STD_TORCH_CHECK(C.sizes().equals(D.sizes()));
    STD_TORCH_CHECK(C.scalar_type() == A.scalar_type(), "All tensors must have the same dtype");
    STD_TORCH_CHECK(C.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(static_cast<int64_t>(idx_C->size()) == C.dim(), 
            "Index vector length must match tensor C dimensions");
  }
  
  TAPP_handle handle;
  TAPP_create_handle(&handle);

  // Get TAPP datatype
  TAPP_datatype dtype = get_tapp_dtype(D);

  // Get extents and strides
  auto extents_A = get_extents(A);
  auto strides_A = get_strides(A);
  auto extents_B = get_extents(B);
  auto strides_B = get_strides(B);
  auto extents_D = get_extents(D);
  auto strides_D = get_strides(D);

  // const std::vector<int64_t>& idx_C_val = idx_C.has_value() ? idx_C.value() : idx_D;

  // TODO do we need contiguity ?
  // torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  // ...

  // Create TAPP tensor info structures
  TAPP_tensor_info info_A, info_B, info_C, info_D;
  TAPP_create_tensor_info(&info_A, handle, dtype, A.dim(), extents_A.data(), strides_A.data());
  TAPP_create_tensor_info(&info_B, handle, dtype, B.dim(), extents_B.data(), strides_B.data());
  TAPP_create_tensor_info(&info_D, handle, dtype, D.dim(), extents_D.data(), strides_D.data());
  if (C.defined()) {
    auto extents_C = get_extents(C);
    auto strides_C = get_strides(C);
    TAPP_create_tensor_info(&info_C, handle, dtype, C.dim(), extents_C.data(), strides_C.data());
  } else {
    // TODO do we always require idx_C == idx_D ? Or in case of C being empty we can avoid this.
    // Use idx_D as default for idx_C
    TAPP_create_tensor_info(&info_C, handle, dtype, D.dim(), extents_D.data(), strides_D.data());
  }

  // Decide elemental operations (conjugate available for complex datatypes)
  TAPP_element_op op_A = TAPP_IDENTITY; // Decide elemental operation for tensor A
  TAPP_element_op op_B = TAPP_IDENTITY; // Decide elemental operation for tensor B
  TAPP_element_op op_C = TAPP_IDENTITY; // Decide elemental operation for tensor C
  TAPP_element_op op_D = TAPP_IDENTITY; // Decide elemental operation for tensor D

  TAPP_prectype prec = TAPP_DEFAULT_PREC; //Choose the calculation precision
  TAPP_tensor_product plan; // Declare the variable that holds the information about the calculation 
  
  TAPP_create_tensor_product(
    &plan, handle,
    op_A, info_A, idx_A.data(),
    op_B, info_B, idx_B.data(),
    op_C, info_C, C.defined() ? idx_C->data() : idx_D.data(),
    op_D, info_D, idx_D.data(),
    prec
  );

  TAPP_executor exec; // Declaration of executor
  TAPP_create_executor(&exec); // Creation of executor
  // int exec_id = 1; // Choose executor
  // exec = (intptr_t)&exec_id; // Assign executor

  // Execute the contraction
  TAPP_status status;
  TAPP_error error = TAPP_execute_product(
    plan, exec, &status,
    (void*)&alpha,
    A.const_data_ptr(),
    B.const_data_ptr(),
    (void*)&beta,
    C.defined() ? C.const_data_ptr() : D.const_data_ptr(),
    (void*)D.mutable_data_ptr()
  );

  // Check for errors
  if (!TAPP_check_success(error)) {
    int msg_len = TAPP_explain_error(error, 0, nullptr);
    std::vector<char> msg_buff(msg_len + 1);
    TAPP_explain_error(error, msg_len + 1, msg_buff.data());
    
    // Cleanup before throwing
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    
    STD_TORCH_CHECK(false, "TAPP contraction failed: ", msg_buff.data());
  }

  // Cleanup
  TAPP_destroy_tensor_product(plan);
  TAPP_destroy_tensor_info(info_A);
  TAPP_destroy_tensor_info(info_B);
  TAPP_destroy_tensor_info(info_C);
  TAPP_destroy_tensor_info(info_D);
  TAPP_destroy_executor(exec);
  TAPP_destroy_handle(handle);
}

// Non-templated wrapper for Python binding (alpha/beta as 0-dim tensors)
void tensor_product_cuda(
  const torch::stable::Tensor& A,
  const torch::stable::Tensor& B,
  const torch::stable::Tensor& C,
  torch::stable::Tensor& D,
  const std::vector<int64_t>& idx_A,
  const std::vector<int64_t>& idx_B,
  const std::optional<std::vector<int64_t>>& idx_C,
  const std::vector<int64_t>& idx_D,
  const torch::stable::Tensor& alpha_t,
  const torch::stable::Tensor& beta_t) {

  STD_TORCH_CHECK(alpha_t.defined() && beta_t.defined(), "alpha/beta must be defined");
  STD_TORCH_CHECK(alpha_t.dim() == 0 && beta_t.dim() == 0, "alpha/beta must be 0-dim tensors");
  STD_TORCH_CHECK(alpha_t.scalar_type() == beta_t.scalar_type(), "alpha/beta dtype must match");

  // currently not available ?
  // stable ABI does not support Scalar.
  //
  // THO_DISPATCH_FLOATING_TYPES(D.scalar_type(), "tensor_product_impl_cuda", [&] {
  //   // scalar_t is the resolved type
  //   tensor_product_impl_cuda<scalar_type>(A, B, C, D, idx_A, idx_B, idx_C, idx_D,
  //                                 static_cast<scalar_type>(alpha),
  //                                 static_cast<scalar_type>(beta));
  // });
  
  auto alpha_d = torch::stable::to(alpha_t, D.scalar_type(), std::nullopt, 
    torch::stable::Device(torch::headeronly::DeviceType::CPU));
  auto beta_d  = torch::stable::to(beta_t, D.scalar_type(), std::nullopt, 
    torch::stable::Device(torch::headeronly::DeviceType::CPU));

  switch (D.scalar_type()) {
      case torch::headeronly::ScalarType::Float: {
          auto alpha = *static_cast<const float*>(alpha_d.const_data_ptr());
          auto beta  = *static_cast<const float*>(beta_d.const_data_ptr());
          tensor_product_impl_cuda<float>(A, B, C, D, idx_A, idx_B, idx_C, idx_D, alpha, beta);
          break;
      }
      case torch::headeronly::ScalarType::Double: {
          auto alpha = *static_cast<const double*>(alpha_d.const_data_ptr());
          auto beta  = *static_cast<const double*>(beta_d.const_data_ptr());
          tensor_product_impl_cuda<double>(A, B, C, D, idx_A, idx_B, idx_C, idx_D, alpha, beta);
          break;
      }
      case torch::headeronly::ScalarType::ComplexFloat: {
          auto alpha = *static_cast<const std::complex<float>*>(alpha_d.const_data_ptr());
          auto beta  = *static_cast<const std::complex<float>*>(beta_d.const_data_ptr());
          tensor_product_impl_cuda<std::complex<float>>(A, B, C, D, idx_A, idx_B, idx_C, idx_D, alpha, beta);
          break;
      }
      case torch::headeronly::ScalarType::ComplexDouble: {
          auto alpha = *static_cast<const std::complex<double>*>(alpha_d.const_data_ptr());
          auto beta  = *static_cast<const std::complex<double>*>(beta_d.const_data_ptr());
          tensor_product_impl_cuda<std::complex<double>>(A, B, C, D, idx_A, idx_B, idx_C, idx_D, alpha, beta);
          break;
      }
      default:
          throw std::runtime_error("Unsupported dtype alpha/beta for TAPP contraction");
  }
}

// Registers CUDA implementation
STABLE_TORCH_LIBRARY_IMPL(tapp_torch, CUDA, m) {
  m.impl("tensor_product", TORCH_BOX(&tensor_product_cuda));
}

} // namespace tapp_torch