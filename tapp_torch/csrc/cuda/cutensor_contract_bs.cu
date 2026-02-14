#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <vector>
#include <algorithm>
#include <complex>

#include <cuda_runtime.h>
#include <cutensor.h>

// Optional NVTX3 support
#if __has_include(<nvtx3/nvtx3.hpp>)
    #include <nvtx3/nvtx3.hpp>
    #define CUBLOCKSPARSE_HAS_NVTX 1
#else
    #define CUBLOCKSPARSE_HAS_NVTX 0
#endif
// NVTX helper macro - no-op when NVTX is not available
#if CUBLOCKSPARSE_HAS_NVTX
    #define NVTX_MARK(msg) nvtx3::mark(msg)
#else
    #define NVTX_MARK(msg) ((void)0)
#endif

// Handle cuTENSOR errors
#define HANDLE_ERROR(x)                                                           \
{                                                                                 \
    const cutensorStatus_t err = (x);                                             \
    if ( err != CUTENSOR_STATUS_SUCCESS )                                         \
    { throw std::runtime_error { std::string { cutensorGetErrorString(err) } }; } \
};

// Handle CUDA errors.
#define HANDLE_CUDA_ERROR(x)                                                  \
{                                                                             \
    const cudaError_t err = (x);                                              \
    if ( err != cudaSuccess )                                                 \
    { throw std::runtime_error { std::string { cudaGetErrorString(err) } }; } \
};

template <typename T>
using cuda_ptr = std::unique_ptr<T,decltype(&cudaFree)>;

template <typename T>
cuda_ptr<T> cuda_alloc( size_t count )
{
    void* result;
    cudaError_t err = cudaMalloc( &result, sizeof(T)*count );
    if ( err != cudaSuccess ) throw std::bad_alloc {};
    else return cuda_ptr<T> { reinterpret_cast<T*>(result), &cudaFree };
}

template <typename T>
using cudaAsync_ptr = std::unique_ptr<T, std::function<void(T*)>>;

template <typename T>
cudaAsync_ptr<T> cuda_async_alloc(size_t count, cudaStream_t stream)
{
    void* result;

    // Use cudaMallocAsync for allocation
    HANDLE_CUDA_ERROR(cudaMallocAsync(&result, sizeof(T) * count, stream));

    auto deleter = [stream](T* ptr) {
        if (ptr) {
            HANDLE_CUDA_ERROR(cudaFreeAsync(ptr, stream));
        }
    };

    return cudaAsync_ptr<T>{reinterpret_cast<T*>(result), deleter};
}

template <typename T>
struct Guard
{
    using destructor = cutensorStatus_t (*)( T );

    T p { nullptr };
    destructor destroy { nullptr };
    ~Guard() { if (p) destroy(p); }
};

namespace tapp_torch {

namespace {

template <typename T>
inline void fill_block_pointers(
  const torch::stable::Tensor& tensor, 
  const std::vector<int64_t>& offsets, 
  std::vector<T*>& ptrs
) {
  T* base_ptr = static_cast<T*>(tensor.mutable_data_ptr());
  std::transform(offsets.begin(), offsets.end(), ptrs.begin(), [base_ptr](int64_t offset) {
    return base_ptr + offset;
  });
};

template <typename T>
inline void fill_block_pointers(
  const torch::stable::Tensor& tensor, 
  const std::vector<int64_t>& offsets, 
  std::vector<const T*>& ptrs
) {
  // const T* base_ptr = tensor.const_data_ptr();
  const T* base_ptr = static_cast<const T*>(tensor.const_data_ptr());
  std::transform(offsets.begin(), offsets.end(), ptrs.begin(), [base_ptr](int64_t offset) {
    return base_ptr + offset;
  });
}

inline cudaDataType_t to_cuda_dtype(torch::headeronly::ScalarType dtype) {
  switch (dtype) {
    case torch::headeronly::ScalarType::Double:
      return CUDA_R_64F;
    case torch::headeronly::ScalarType::ComplexDouble:
      return CUDA_C_64F;
    case torch::headeronly::ScalarType::Float:
      return CUDA_R_32F;
    case torch::headeronly::ScalarType::ComplexFloat:
      return CUDA_C_32F;
    default:
      throw std::runtime_error{"Unsupported dtype."};
  }
}

inline cutensorComputeDescriptor_t to_cuda_compute_desc(torch::headeronly::ScalarType dtype) {
  switch (dtype) {
    case torch::headeronly::ScalarType::Double:
    case torch::headeronly::ScalarType::ComplexDouble:
      return CUTENSOR_COMPUTE_DESC_64F;
    case torch::headeronly::ScalarType::Float:
    case torch::headeronly::ScalarType::ComplexFloat:
      return CUTENSOR_COMPUTE_DESC_32F;
    default:
      throw std::runtime_error{"Unsupported dtype."};
  }
}

void wrap_BlockSparseTensorDescriptor(
    cutensorHandle_t& handle,
    const std::vector<int64_t>& numSectionsPerMode, // number of sections per mode     
    const std::vector<int64_t>& sectionExtents,     // extents of the sections in modes of the tensor, 
                                                    // linearized from first mode first section to last mode last section
    const std::vector<int64_t>& blocks,   // coordinates of non-zero blocks 
    const std::vector<int64_t>& strides,  
    cudaDataType_t dataType,
    cutensorBlockSparseTensorDescriptor_t& desc
    // BlockSparseTensorGuard& descGuard
) {
  std::vector<uint32_t> nSectionsPerMode_u32(numSectionsPerMode.begin(), numSectionsPerMode.end());
  std::vector<int32_t> blocks_u32(blocks.begin(), blocks.end());
  uint32_t nModes = static_cast<uint32_t>(numSectionsPerMode.size());
  uint64_t nBlocks = static_cast<uint64_t>(blocks.size() / numSectionsPerMode.size());
  HANDLE_ERROR(cutensorCreateBlockSparseTensorDescriptor(
        handle, &desc,
        nModes, nBlocks, 
        nSectionsPerMode_u32.data(), 
        sectionExtents.data(),
        blocks_u32.data(), 
        strides.data(), dataType
  ));
}

}

using ModeType   = int32_t;
using ExtentType = int64_t;
using StrideType = int64_t;

// D_{modeD} <- opA(A_{modeA})opB(B_{modeB})+opC(C_{modeC})
template <typename scalar_t>
void tensor_product_bs_cuda_impl(
    const std::vector<const scalar_t *>& A,               
    const std::vector<const scalar_t *>& B,
    const std::optional<std::vector<const scalar_t *>>& C,
    const std::vector<scalar_t *>& D,
    const std::vector<int64_t>& a_modes,              // mode labels of A
    const std::vector<int64_t>& a_numSectionsPerMode, // number of sections per mode of A     
    const std::vector<int64_t>& a_sectionExtents,     // extents of the sections in modes of A
    const std::vector<int64_t>& a_blocks,             // Coordinates of the non-zero blocks in the tensor, which are specified as a vector of indices
                                                      // with respect to sectionExtents already serialized into 1D, i.e.
                                                      // { { x0_0, x0_1, ..., x0_#modes-1 },
                                                      //   { x1_0,       ..., x1_#modes-1 },
                                                      //   ...                               } is given as 
                                                      // { x0_0, x0_1, ..., x0_#modes-1, x1_0, ..., x1_#modes-1, ... }
    const std::vector<int64_t>& a_strides,  
    const std::vector<int64_t>& b_modes,
    const std::vector<int64_t>& b_numSectionsPerMode, 
    const std::vector<int64_t>& b_sectionExtents,
    const std::vector<int64_t>& b_blocks,
    const std::vector<int64_t>& b_strides,  
    const std::optional<std::vector<int64_t>>& c_modes,
    const std::optional<std::vector<int64_t>>& c_numSectionsPerMode, 
    const std::optional<std::vector<int64_t>>& c_sectionExtents,
    const std::optional<std::vector<int64_t>>& c_blocks,
    const std::optional<std::vector<int64_t>>& c_strides,
    const std::vector<int64_t>& d_modes,
    const std::vector<int64_t>& d_numSectionsPerMode, 
    const std::vector<int64_t>& d_sectionExtents,
    const std::vector<int64_t>& d_blocks,
    const std::vector<int64_t>& d_strides,
    const scalar_t alpha,
    const scalar_t beta,
    cutensorComputeDescriptor_t computeDesc,
    cudaDataType_t dtype,
    cudaStream_t* stream_ptr = nullptr
) {
  const char* env = std::getenv("TAPP_LOG_LEVEL");
  int tapp_log_level = (env) ? std::atoi(env) : 0;
       
  cutensorHandle_t handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  cudaStream_t stream;
  if (stream_ptr) {
    stream = *stream_ptr;  
  } else {
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
  }

  // cast *_modes to int32_t
  std::vector<int32_t> a_modes_32(a_modes.begin(), a_modes.end());
  std::vector<int32_t> b_modes_32(b_modes.begin(), b_modes.end());
  std::vector<int32_t> d_modes_32(d_modes.begin(), d_modes.end());
  std::vector<int32_t> c_modes_32;

  // Get descriptors
  cutensorBlockSparseTensorDescriptor_t a_desc, b_desc, c_desc, d_desc;
  wrap_BlockSparseTensorDescriptor(
    handle, a_numSectionsPerMode, a_sectionExtents, a_blocks, a_strides, 
    dtype, a_desc);
  wrap_BlockSparseTensorDescriptor(
    handle, b_numSectionsPerMode, b_sectionExtents, b_blocks, b_strides, 
    dtype, b_desc);
  wrap_BlockSparseTensorDescriptor(
    handle, d_numSectionsPerMode, d_sectionExtents, d_blocks, d_strides, 
    dtype, d_desc);
  if (C.has_value()) {
    STD_TORCH_CHECK(c_modes.has_value() && c_numSectionsPerMode.has_value() 
      && c_sectionExtents.has_value() && c_blocks.has_value() && c_strides.has_value(),
      "If C is defined, all of c_modes, c_numSectionsPerMode, c_sectionExtents, c_blocks, and c_strides must be provided");
    c_modes_32 = std::vector<int32_t>(c_modes->begin(), c_modes->end());
    wrap_BlockSparseTensorDescriptor(
        handle, c_numSectionsPerMode.value(), c_sectionExtents.value(), 
        c_blocks.value(), c_strides.value(), dtype, c_desc);
  } else {
    c_modes_32 = d_modes_32;
    c_desc = d_desc;
  }

  /*******************************
   * Block-sparse Contraction.   *
   *******************************/

  // Create contraction descriptor
  cutensorOperationDescriptor_t contractionDesc;
  // HANDLE_ERROR(cutensorCreateBlockSparseContraction(
  //     handle, &contractionDesc,
  //     a_desc, a_modes_32.data(), CUTENSOR_OP_IDENTITY,
  //     b_desc, b_modes_32.data(), CUTENSOR_OP_IDENTITY,
  //     c_desc, c_modes_32.data(), CUTENSOR_OP_IDENTITY,
  //     d_desc, d_modes_32.data(),
  //     computeDesc
  // ));
  // NOTE See https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcreateblocksparsecontractiondescriptor
  //      for current API limitations 
  HANDLE_ERROR(cutensorCreateBlockSparseContraction(
      handle, &contractionDesc,
      a_desc, a_modes_32.data(), CUTENSOR_OP_IDENTITY,
      b_desc, b_modes_32.data(), CUTENSOR_OP_IDENTITY,
      d_desc, c_modes_32.data(), CUTENSOR_OP_IDENTITY,
      d_desc, d_modes_32.data(),
      computeDesc
  ));
    
  // Create plan preference (using default settings here)
  cutensorPlanPreference_t planPref = nullptr;
  // const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
  // const cutensorJitMode_t jitMode = CUTENSOR_JIT_MODE_NONE;
  // HANDLE_ERROR(cutensorCreatePlanPreference(handle,&planPref,algo,jitMode));
  // Guard<cutensorPlanPreference_t> guardPlanPref { planPref, &cutensorDestroyPlanPreference };
  
  // Query workspace
  uint64_t workspaceSizeEstimate; // in bytes
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  HANDLE_ERROR(cutensorEstimateWorkspaceSize(
      handle, contractionDesc, planPref, workspacePref, &workspaceSizeEstimate
  ));

  // Create plan
  cutensorPlan_t plan;
  HANDLE_ERROR(cutensorCreatePlan(
      handle, &plan, contractionDesc, planPref, workspaceSizeEstimate
  ));

  // See https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcontract 
  // for details on workspace allocation alignment requirements.
  auto workspace = cuda_async_alloc<char>(workspaceSizeEstimate, stream);

  if (tapp_log_level>5) NVTX_MARK( "tapp_torch::cutensorBlockSparseContract" );
  HANDLE_ERROR(cutensorBlockSparseContract(handle, plan,
              (const void*) &alpha, (const void *const *) A.data(), (const void *const *) B.data(),
              (const void*) &beta,  (const void *const *) (C.has_value() ? C.value().data() : D.data()), 
              (void *const *) D.data(), 
              (void*) workspace.get(), workspaceSizeEstimate, stream));

  HANDLE_ERROR(cutensorDestroyBlockSparseTensorDescriptor(a_desc));
  HANDLE_ERROR(cutensorDestroyBlockSparseTensorDescriptor(b_desc));
  HANDLE_ERROR(cutensorDestroyBlockSparseTensorDescriptor(d_desc));
  if (C.has_value()) {
    HANDLE_ERROR(cutensorDestroyBlockSparseTensorDescriptor(c_desc));
  }
  HANDLE_ERROR(cutensorDestroyPlan(plan));
  if (stream_ptr) {
    // Stream was provided by caller, do not destroy
  } else {
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
  }
  HANDLE_ERROR(cutensorDestroy(handle));

  // return EXIT_SUCCESS;
}
// }
// catch ( std::exception &ex )
// {
//   std::cerr << "Exception. Exiting." << std::endl;
//   std::cerr << ex.what() << std::endl;
//   return EXIT_FAILURE;
// }
// catch ( ... )
// {
//   std::cerr << "Unknown exception. Exiting." << std::endl;
//   return EXIT_FAILURE;
// }


// NOTE Assume blocks are stored in contiguous memory associated with a single dense tensor.
//      Blocks as ArrayRef[Tensor], i.e. each block backed by an independent tensor can also be supported in the future.
void tensor_product_bs_cuda(
    const torch::stable::Tensor& A,         // 1D 
    const torch::stable::Tensor& B,
    const torch::stable::Tensor& C,
    torch::stable::Tensor& D,
    const std::vector<int64_t>& a_modes,              // mode labels of A
    const std::vector<int64_t>& a_numSectionsPerMode, // number of sections per mode of A     
    const std::vector<int64_t>& a_sectionExtents,     // extents of the sections in modes of A
    const std::vector<int64_t>& a_blocks,             // Coordinates of the non-zero blocks in the tensor, which are specified as a vector of indices
                                                      // with respect to sectionExtents already serialized into 1D, i.e.
                                                      // { { x0_0, x0_1, ..., x0_#modes-1 },
                                                      //   { x1_0,       ..., x1_#modes-1 },
                                                      //   ...                               } is given as 
                                                      // { x0_0, x0_1, ..., x0_#modes-1, x1_0, ..., x1_#modes-1, ... }
    const std::vector<int64_t>& a_strides,  
    const std::vector<int64_t>& a_offsets,  // offsets of the blocks in the flattened storage of a
    const std::vector<int64_t>& b_modes,
    const std::vector<int64_t>& b_numSectionsPerMode, 
    const std::vector<int64_t>& b_sectionExtents,
    const std::vector<int64_t>& b_blocks,
    const std::vector<int64_t>& b_strides,  
    const std::vector<int64_t>& b_offsets,
    const std::optional<std::vector<int64_t>>& c_modes,
    const std::optional<std::vector<int64_t>>& c_numSectionsPerMode, 
    const std::optional<std::vector<int64_t>>& c_sectionExtents,
    const std::optional<std::vector<int64_t>>& c_blocks,
    const std::optional<std::vector<int64_t>>& c_strides,
    const std::optional<std::vector<int64_t>>& c_offsets,
    const std::vector<int64_t>& d_modes,
    const std::vector<int64_t>& d_numSectionsPerMode, 
    const std::vector<int64_t>& d_sectionExtents,
    const std::vector<int64_t>& d_blocks,
    const std::vector<int64_t>& d_strides,  
    const std::vector<int64_t>& d_offsets,
    const torch::stable::Tensor& alpha_t,
    const torch::stable::Tensor& beta_t
) {
  const char* env = std::getenv("TAPP_LOG_LEVEL");
  int tapp_log_level = (env) ? std::atoi(env) : 0;

  STD_TORCH_CHECK(A.dim() == 1, "Tensor A must be 1D.");
  STD_TORCH_CHECK(B.dim() == 1, "Tensor B must be 1D.");
  STD_TORCH_CHECK(D.dim() == 1, "Tensor D must be 1D.");
  STD_TORCH_CHECK(A.scalar_type() == B.scalar_type() && A.scalar_type() == D.scalar_type(), "All tensors must have the same dtype.");
  
  STD_TORCH_CHECK(A.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(B.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(D.device().type() == torch::headeronly::DeviceType::CUDA);

  STD_TORCH_CHECK(alpha_t.defined() && beta_t.defined(), "alpha/beta must be defined");
  STD_TORCH_CHECK(alpha_t.dim() == 0 && beta_t.dim() == 0, "alpha/beta must be 0-dim tensors");
  STD_TORCH_CHECK(alpha_t.scalar_type() == beta_t.scalar_type(), "alpha/beta dtype must match");

  if (C.defined()) {
    STD_TORCH_CHECK(C.dim() == 1, "Tensor C must be 1D.");
    STD_TORCH_CHECK(A.scalar_type() == C.scalar_type(), "Tensor C must have the same dtype as D (add A and B)");
    STD_TORCH_CHECK(C.device().type() == torch::headeronly::DeviceType::CUDA);
  } else {
    // Check if beta is 0
    // Accept both scalar and rank-1 tensor with a single element
  }

  // NOTE https://docs.pytorch.org/cppdocs/stable.html#getting-the-current-cuda-stream
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
    aoti_torch_get_current_cuda_stream(D.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  STD_TORCH_CHECK(!C.defined() || (c_modes.has_value() && c_blocks.has_value() && c_offsets.has_value()),
      "If C is defined, all of c_modes, c_blocks, and c_offsets must be provided");

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

  auto l_tensor_product_bs_cuda_impl = [&](auto alpha, auto beta) {
    using scalar_t = typename std::remove_cv<decltype(alpha)>::type; // base type

    // Prepare device pointers to blocks
    uint64_t a_nblocks = a_blocks.size() / a_modes.size();
    uint64_t b_nblocks = b_blocks.size() / b_modes.size();
    uint64_t d_nblocks = d_blocks.size() / d_modes.size();
    std::vector<const scalar_t*> a(a_nblocks);
    std::vector<const scalar_t*> b(b_nblocks);
    std::vector<scalar_t*> d(d_nblocks);
    std::optional<std::vector<const scalar_t*>> c;

    fill_block_pointers<scalar_t>(A, a_offsets, a);
    fill_block_pointers<scalar_t>(B, b_offsets, b);
    fill_block_pointers<scalar_t>(D, d_offsets, d);
    if (C.defined()) {
      uint64_t c_nblocks = c_blocks.value().size() / c_modes.value().size();
      std::vector<const scalar_t*> c_ptrs(c_nblocks);
      fill_block_pointers<scalar_t>(C, c_offsets.value(), c_ptrs);
      c.emplace(std::move(c_ptrs));
    } else {
      c = std::nullopt;
    }

    tensor_product_bs_cuda_impl<scalar_t>(
        a, b, c, d,
        a_modes, a_numSectionsPerMode, a_sectionExtents, a_blocks, a_strides,
        b_modes, b_numSectionsPerMode, b_sectionExtents, b_blocks, b_strides,
        c_modes, c_numSectionsPerMode, c_sectionExtents, c_blocks, c_strides,
        d_modes, d_numSectionsPerMode, d_sectionExtents, d_blocks, d_strides,
        alpha, beta, to_cuda_compute_desc(D.scalar_type()), to_cuda_dtype(D.scalar_type()), &stream
    );
  };

// Use the lambda for each scalar type
switch (D.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
        auto alpha = *static_cast<const float*>(alpha_d.const_data_ptr());
        auto beta  = *static_cast<const float*>(beta_d.const_data_ptr());
        l_tensor_product_bs_cuda_impl(alpha, beta);
        break;
    }
    case torch::headeronly::ScalarType::Double: {
        auto alpha = *static_cast<const double*>(alpha_d.const_data_ptr());
        auto beta  = *static_cast<const double*>(beta_d.const_data_ptr());
        l_tensor_product_bs_cuda_impl(alpha, beta);
        break;
    }
    case torch::headeronly::ScalarType::ComplexFloat: {
        auto alpha = *static_cast<const std::complex<float>*>(alpha_d.const_data_ptr());
        auto beta  = *static_cast<const std::complex<float>*>(beta_d.const_data_ptr());
        l_tensor_product_bs_cuda_impl(alpha, beta);
        break;
    }
    case torch::headeronly::ScalarType::ComplexDouble: {
        auto alpha = *static_cast<const std::complex<double>*>(alpha_d.const_data_ptr());
        auto beta  = *static_cast<const std::complex<double>*>(beta_d.const_data_ptr());
        l_tensor_product_bs_cuda_impl(alpha, beta);
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype alpha/beta for TAPP contraction");
}
}

// Registers CUDA implementation
STABLE_TORCH_LIBRARY_IMPL(tapp_torch, CUDA, m) {
  m.impl("tensor_product_bs", TORCH_BOX(&tensor_product_bs_cuda));
}

} // namespace tapp_torch