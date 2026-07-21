#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cerrno>
#include <iostream>
#include <vector>
#include <algorithm>
#include <complex>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cutensor.h>

// Optional NVTX support — prefer C++ API, fall back to legacy C API
#if __has_include(<nvtx3/nvtx3.hpp>)
    #include <nvtx3/nvtx3.hpp>
    #define CUBLOCKSPARSE_HAS_NVTX 2
#elif __has_include(<nvtx3/nvToolsExt.h>)
    #include <nvtx3/nvToolsExt.h>
    #define CUBLOCKSPARSE_HAS_NVTX 1
#else
    #define CUBLOCKSPARSE_HAS_NVTX 0
#endif
// NVTX helper macro - no-op when NVTX is not available
#if CUBLOCKSPARSE_HAS_NVTX == 2
    #define NVTX_MARK(msg) nvtx3::mark(msg)
#elif CUBLOCKSPARSE_HAS_NVTX == 1
    #define NVTX_MARK(msg) nvtxMarkA(msg)
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

// Returns the cuTENSOR handle for the current CUDA device, creating one if none exists yet.
// One handle is shared across all threads for a given device. 
// When a device switch is detected a fresh handle is created for the new device.
cutensorHandle_t get_cutensor_handle() {
  struct HandleMap {
    std::mutex mutex;
    std::unordered_map<int, cutensorHandle_t> map;
    ~HandleMap() {
      for (auto& [dev, h] : map)
        if (h) cutensorDestroy(h);
    }
  };
  static HandleMap handles;

  int device;
  HANDLE_CUDA_ERROR(cudaGetDevice(&device));

  std::lock_guard<std::mutex> lock(handles.mutex);
  auto it = handles.map.find(device);
  if (it != handles.map.end())
    return it->second;

  cutensorHandle_t handle{};
  HANDLE_ERROR(cutensorCreate(&handle));
  handles.map[device] = handle;
  return handle;
}

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

struct DescriptorKey {
  std::vector<int64_t> numSectionsPerMode;
  std::vector<int64_t> sectionExtents;
  std::vector<int64_t> blocks;
  std::vector<int64_t> strides;
  cudaDataType_t dataType;
  bool operator==(const DescriptorKey& o) const {
    return numSectionsPerMode == o.numSectionsPerMode
        && sectionExtents    == o.sectionExtents
        && blocks            == o.blocks
        && strides           == o.strides
        && dataType          == o.dataType;
  }
};

struct DescriptorKeyHash {
  size_t operator()(const DescriptorKey& k) const {
    auto combine = [](size_t seed, size_t h) -> size_t {
      return seed ^ (h + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    };
    auto hash_vec = [&](const std::vector<int64_t>& v) -> size_t {
      size_t h = v.size();
      for (auto x : v) h = combine(h, std::hash<int64_t>{}(x));
      return h;
    };
    size_t h = 0;
    h = combine(h, hash_vec(k.numSectionsPerMode));
    h = combine(h, hash_vec(k.sectionExtents));
    h = combine(h, hash_vec(k.blocks));
    h = combine(h, hash_vec(k.strides));
    h = combine(h, std::hash<int>{}(static_cast<int>(k.dataType)));
    return h;
  }
};

// Descriptors encode only tensor metadata and are safe to reuse across handle instances.
// Ownership stays in the cache; callers must NOT call cutensorDestroyBlockSparseTensorDescriptor.
// max_size=0 means unlimited. Eviction policy: LRU.
// A single contraction needs up to 4 descriptors (A, B, C, D), so the minimum
// enforced capacity is MIN_CACHE_SIZE when a finite limit is requested.
class BlockSparseDescriptorCache {
public:
  static constexpr size_t MIN_CACHE_SIZE = 4;

  explicit BlockSparseDescriptorCache(size_t max_size)
      : max_size_(clamp_cache_size(max_size)) {
    const char* env = std::getenv("TAPP_LOG_LEVEL");
    log_level_ = env ? std::atoi(env) : 0;
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] desc_cache: created max_size="
                << (max_size_ == 0 ? "unlimited" : std::to_string(max_size_)) << "\n";
  }

  void get_or_create(
      cutensorHandle_t& handle,
      const std::vector<int64_t>& numSectionsPerMode,
      const std::vector<int64_t>& sectionExtents,
      const std::vector<int64_t>& blocks,
      const std::vector<int64_t>& strides,
      cudaDataType_t dataType,
      cutensorBlockSparseTensorDescriptor_t& desc
  ) {
    DescriptorKey key{numSectionsPerMode, sectionExtents, blocks, strides, dataType};
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = map_.find(key);
    if (it != map_.end()) {
      ++hits_;
      lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_it);
      desc = it->second.desc;
      if (log_level_ >= 6)
        std::cout << "[tapp_torch] desc_cache: HIT  size=" << map_.size()
                  << " hits=" << hits_ << " misses=" << misses_ << "\n";
      return;
    }

    ++misses_;
    if (max_size_ > 0 && map_.size() >= max_size_)
      evict_lru_locked();

    uint32_t nModes  = static_cast<uint32_t>(numSectionsPerMode.size());
    uint64_t nBlocks = nModes ? static_cast<uint64_t>(blocks.size() / nModes) : 0;
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] desc_cache: MISS nModes=" << nModes
                << " nBlocks=" << nBlocks
                << " -> creating (size=" << map_.size() + 1 << ")\n";

    std::vector<uint32_t> nSections_u32(numSectionsPerMode.begin(), numSectionsPerMode.end());
    std::vector<int32_t>  blocks_i32(blocks.begin(), blocks.end());
    HANDLE_ERROR(cutensorCreateBlockSparseTensorDescriptor(
        handle, &desc,
        nModes, nBlocks,
        nSections_u32.data(),
        sectionExtents.data(),
        blocks_i32.data(),
        strides.data(), dataType
    ));
    auto map_it = map_.emplace(std::move(key), Entry{desc, lru_list_.end()}).first;
    lru_list_.push_front(&map_it->first);
    map_it->second.lru_it = lru_list_.begin();
  }

  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] desc_cache: CLEAR (was " << map_.size() << " entries)\n";
    cutensorStatus_t last_err = CUTENSOR_STATUS_SUCCESS;
    for (auto& [key, entry] : map_) {
      auto err = cutensorDestroyBlockSparseTensorDescriptor(entry.desc);
      if (err != CUTENSOR_STATUS_SUCCESS) last_err = err;
    }
    map_.clear();
    lru_list_.clear();
    hits_ = misses_ = 0;
    if (last_err != CUTENSOR_STATUS_SUCCESS)
      throw std::runtime_error{std::string{cutensorGetErrorString(last_err)}};
  }

  void set_max_size(size_t n) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    max_size_ = clamp_cache_size(n);
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] desc_cache: RESIZE max_size="
                << (max_size_ == 0 ? "unlimited" : std::to_string(max_size_))
                << " current=" << map_.size() << "\n";
    if (max_size_ == 0) return;
    while (map_.size() > max_size_)
      evict_lru_locked();
  }

  std::pair<uint64_t, uint64_t> stats() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return {hits_, misses_};
  }

  std::pair<size_t, size_t> size_info() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    size_t bytes = 0;
    for (const auto& [key, entry] : map_)
      bytes += sizeof(int64_t) * (key.numSectionsPerMode.size() + key.sectionExtents.size()
                                 + key.blocks.size() + key.strides.size())
             + sizeof(cudaDataType_t) + sizeof(cutensorBlockSparseTensorDescriptor_t);
    return {map_.size(), bytes};
  }

  ~BlockSparseDescriptorCache() noexcept(false) {
    cutensorStatus_t last_err = CUTENSOR_STATUS_SUCCESS;
    for (auto& [key, entry] : map_) {
      auto err = cutensorDestroyBlockSparseTensorDescriptor(entry.desc);
      if (err != CUTENSOR_STATUS_SUCCESS) last_err = err;
    }
    if (last_err != CUTENSOR_STATUS_SUCCESS)
      throw std::runtime_error{std::string{cutensorGetErrorString(last_err)}};
  }

private:
  struct Entry {
    cutensorBlockSparseTensorDescriptor_t       desc;
    std::list<const DescriptorKey*>::iterator   lru_it;
  };

  static size_t clamp_cache_size(size_t n) {
    return (n > 0 && n < MIN_CACHE_SIZE) ? MIN_CACHE_SIZE : n;
  }

  void evict_lru_locked() {
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] desc_cache: EVICT LRU (size=" << map_.size() - 1 << ")\n";
    auto it = map_.find(*lru_list_.back());
    auto desc = it->second.desc;
    map_.erase(it);
    lru_list_.pop_back();
    HANDLE_ERROR(cutensorDestroyBlockSparseTensorDescriptor(desc));
  }

  int      log_level_ = 0;
  size_t   max_size_;
  std::unordered_map<DescriptorKey, Entry, DescriptorKeyHash> map_;
  std::list<const DescriptorKey*>                             lru_list_;
  mutable std::shared_mutex                                   mutex_;
  uint64_t hits_   = 0;
  uint64_t misses_ = 0;
};

inline size_t parse_cache_size_env(const char* var, size_t default_val, size_t min_size) {
  const char* env = std::getenv(var);
  if (!env) return default_val;

  char* end;
  errno = 0;
  long val = std::strtol(env, &end, 10);

  if (end == env || *end != '\0' || errno != 0 || val < 0) {
    std::cerr << "[tapp_torch] Warning: " << var << "=\"" << env
              << "\" is not a valid non-negative integer; using default " << default_val << ".\n";
    return default_val;
  }

  size_t n = static_cast<size_t>(val);
  if (n > 0 && n < min_size) {
    std::cerr << "[tapp_torch] Warning: " << var << "=" << n
              << " is below the minimum of " << min_size << "; using " << min_size << ".\n";
    return min_size;
  }

  return n;
}

BlockSparseDescriptorCache g_descriptor_cache{
    parse_cache_size_env("TAPP_DESCRIPTOR_CACHE_SIZE", 1024,
                         BlockSparseDescriptorCache::MIN_CACHE_SIZE)};

// ── Contraction descriptor + plan cache ──────────────────────────────────
// Key covers all inputs to cutensorCreateBlockSparseContraction +
// cutensorEstimateWorkspaceSize.  The c position always receives d_desc due
// to a cuTENSOR API limitation, so c_key is always the D descriptor key.
struct ContractionPlanKey {
  DescriptorKey        a_key;  std::vector<int32_t> a_modes;
  DescriptorKey        b_key;  std::vector<int32_t> b_modes;
  DescriptorKey        c_key;  std::vector<int32_t> c_modes;
  DescriptorKey        d_key;  std::vector<int32_t> d_modes;
  cutensorComputeDescriptor_t  computeDesc;
  cutensorWorksizePreference_t workspacePref;
  cutensorAlgo_t               algo;
  cutensorJitMode_t            jitMode;

  bool operator==(const ContractionPlanKey& o) const {
    return a_key == o.a_key && a_modes == o.a_modes
        && b_key == o.b_key && b_modes == o.b_modes
        && c_key == o.c_key && c_modes == o.c_modes
        && d_key == o.d_key && d_modes == o.d_modes
        && computeDesc   == o.computeDesc
        && workspacePref == o.workspacePref
        && algo    == o.algo
        && jitMode == o.jitMode;
  }
};

struct ContractionPlanKeyHash {
  size_t operator()(const ContractionPlanKey& k) const {
    auto combine = [](size_t seed, size_t h) -> size_t {
      return seed ^ (h + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    };
    auto hash_modes = [&](const std::vector<int32_t>& v) -> size_t {
      size_t h = v.size();
      for (auto x : v) h = combine(h, std::hash<int32_t>{}(x));
      return h;
    };
    DescriptorKeyHash dkh;
    size_t h = 0;
    h = combine(h, dkh(k.a_key)); h = combine(h, hash_modes(k.a_modes));
    h = combine(h, dkh(k.b_key)); h = combine(h, hash_modes(k.b_modes));
    h = combine(h, dkh(k.c_key)); h = combine(h, hash_modes(k.c_modes));
    h = combine(h, dkh(k.d_key)); h = combine(h, hash_modes(k.d_modes));
    h = combine(h, std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(k.computeDesc)));
    h = combine(h, std::hash<int>{}(static_cast<int>(k.workspacePref)));
    h = combine(h, std::hash<int>{}(static_cast<int>(k.algo)));
    h = combine(h, std::hash<int>{}(static_cast<int>(k.jitMode)));
    return h;
  }
};

class BlockSparseContractionPlanCache {
public:
  static constexpr size_t MIN_CACHE_SIZE = 1;

  explicit BlockSparseContractionPlanCache(size_t max_size)
      : max_size_(clamp(max_size)) {
    const char* env = std::getenv("TAPP_LOG_LEVEL");
    log_level_ = env ? std::atoi(env) : 0;
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] plan_cache: created max_size="
                << (max_size_ == 0 ? "unlimited" : std::to_string(max_size_)) << "\n";
  }

  void get_or_create(
      cutensorHandle_t&                     handle,
      const ContractionPlanKey&             key,
      const cutensorBlockSparseTensorDescriptor_t a_desc,
      const cutensorBlockSparseTensorDescriptor_t b_desc,
      const cutensorBlockSparseTensorDescriptor_t c_desc,
      const cutensorBlockSparseTensorDescriptor_t d_desc,
      cutensorOperationDescriptor_t&        out_contractionDesc,
      uint64_t&                             out_workspaceSizeEstimate,
      cutensorPlan_t&                       out_plan
  ) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = map_.find(key);
    if (it != map_.end()) {
      ++hits_;
      lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_it);
      out_contractionDesc       = it->second.cached.contractionDesc;
      out_workspaceSizeEstimate = it->second.cached.workspaceSizeEstimate;
      out_plan                  = it->second.cached.plan;
      if (log_level_ >= 6)
        std::cout << "[tapp_torch] plan_cache: HIT  size=" << map_.size()
                  << " hits=" << hits_ << " misses=" << misses_ << "\n";
      return;
    }

    ++misses_;
    if (max_size_ > 0 && map_.size() >= max_size_)
      evict_lru_locked();

    cutensorOperationDescriptor_t contractionDesc;
    HANDLE_ERROR(cutensorCreateBlockSparseContraction(
        handle, &contractionDesc,
        a_desc, key.a_modes.data(), CUTENSOR_OP_IDENTITY,
        b_desc, key.b_modes.data(), CUTENSOR_OP_IDENTITY,
        c_desc, key.c_modes.data(), CUTENSOR_OP_IDENTITY,
        d_desc, key.d_modes.data(),
        key.computeDesc
    ));

    // CUTENSOR_OPERATION_DESCRIPTOR_BLOCKSPARSE_REPRODUCIBLE was introduced in
    // cuTENSOR 2.7.0 (CUTENSOR_VERSION == 20700).
#if defined(CUTENSOR_VERSION) && CUTENSOR_VERSION >= 20700
    const char* env_reproducible = std::getenv("CUTENSOR_BLOCKSPARSE_REPRODUCIBLE");
    int32_t reproducible = env_reproducible ? std::atoi(env_reproducible) : 0;
    if ((log_level_ >= 6) && (reproducible != 0))
      std::cout << "[tapp_torch] plan_cache: Setting CUTENSOR_OPERATION_DESCRIPTOR_BLOCKSPARSE_REPRODUCIBLE="
                << reproducible << "\n";
    HANDLE_ERROR(cutensorOperationDescriptorSetAttribute(
        handle, contractionDesc,
        CUTENSOR_OPERATION_DESCRIPTOR_BLOCKSPARSE_REPRODUCIBLE,
        &reproducible,
        sizeof(reproducible)
    ));
#endif

    cutensorPlanPreference_t planPref = nullptr;
    // HANDLE_ERROR(cutensorCreatePlanPreference(handle, &planPref, key.algo, key.jitMode));

    uint64_t workspaceSizeEstimate;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(
        handle, contractionDesc, planPref, key.workspacePref, &workspaceSizeEstimate
    ));

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(
        handle, &plan, contractionDesc, planPref, workspaceSizeEstimate
    ));

    // HANDLE_ERROR(cutensorDestroyPlanPreference(planPref));

    if (log_level_ >= 6)
      std::cout << "[tapp_torch] plan_cache: MISS aModes=" << key.a_modes.size()
                << " bModes=" << key.b_modes.size()
                << " cModes=" << key.c_modes.size()
                << " dModes=" << key.d_modes.size()
                << " workspace=" << workspaceSizeEstimate << "B"
                << " -> creating (size=" << map_.size() + 1 << ")\n";

    auto map_it = map_.emplace(std::move(key),
        Entry{Cached{contractionDesc, workspaceSizeEstimate, plan}, lru_list_.end()}).first;
    lru_list_.push_front(&map_it->first);
    map_it->second.lru_it = lru_list_.begin();

    out_contractionDesc       = contractionDesc;
    out_workspaceSizeEstimate = workspaceSizeEstimate;
    out_plan                  = plan;
  }

  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] plan_cache: CLEAR (was " << map_.size() << " entries)\n";
    cutensorStatus_t last_err = CUTENSOR_STATUS_SUCCESS;
    for (auto& [key, entry] : map_) {
      auto e1 = cutensorDestroyOperationDescriptor(entry.cached.contractionDesc);
      auto e2 = cutensorDestroyPlan(entry.cached.plan);
      if (e1 != CUTENSOR_STATUS_SUCCESS) last_err = e1;
      if (e2 != CUTENSOR_STATUS_SUCCESS) last_err = e2;
    }
    map_.clear();
    lru_list_.clear();
    hits_ = misses_ = 0;
    if (last_err != CUTENSOR_STATUS_SUCCESS)
      throw std::runtime_error{std::string{cutensorGetErrorString(last_err)}};
  }

  void set_max_size(size_t n) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    max_size_ = clamp(n);
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] plan_cache: RESIZE max_size="
                << (max_size_ == 0 ? "unlimited" : std::to_string(max_size_))
                << " current=" << map_.size() << "\n";
    if (max_size_ == 0) return;
    while (map_.size() > max_size_)
      evict_lru_locked();
  }

  std::pair<uint64_t, uint64_t> stats() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return {hits_, misses_};
  }

  std::pair<size_t, size_t> size_info() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    size_t bytes = 0;
    for (const auto& [key, entry] : map_) {
      auto dkey_bytes = [](const DescriptorKey& dk) {
        return sizeof(int64_t) * (dk.numSectionsPerMode.size() + dk.sectionExtents.size()
                                  + dk.blocks.size() + dk.strides.size())
               + sizeof(cudaDataType_t);
      };
      auto modes_bytes = [](const std::vector<int32_t>& v) {
        return sizeof(int32_t) * v.size();
      };
      bytes += dkey_bytes(key.a_key) + modes_bytes(key.a_modes)
             + dkey_bytes(key.b_key) + modes_bytes(key.b_modes)
             + dkey_bytes(key.c_key) + modes_bytes(key.c_modes)
             + dkey_bytes(key.d_key) + modes_bytes(key.d_modes)
             + sizeof(cutensorComputeDescriptor_t) + sizeof(cutensorWorksizePreference_t)
             + sizeof(cutensorOperationDescriptor_t) + sizeof(cutensorPlan_t);
    }
    return {map_.size(), bytes};
  }

  ~BlockSparseContractionPlanCache() noexcept(false) {
    cutensorStatus_t last_err = CUTENSOR_STATUS_SUCCESS;
    for (auto& [key, entry] : map_) {
      auto e1 = cutensorDestroyOperationDescriptor(entry.cached.contractionDesc);
      auto e2 = cutensorDestroyPlan(entry.cached.plan);
      if (e1 != CUTENSOR_STATUS_SUCCESS) last_err = e1;
      if (e2 != CUTENSOR_STATUS_SUCCESS) last_err = e2;
    }
    if (last_err != CUTENSOR_STATUS_SUCCESS)
      throw std::runtime_error{std::string{cutensorGetErrorString(last_err)}};
  }

private:
  struct Cached {
    cutensorOperationDescriptor_t contractionDesc;
    uint64_t                      workspaceSizeEstimate;
    cutensorPlan_t                plan;
  };
  struct Entry {
    Cached                                            cached;
    std::list<const ContractionPlanKey*>::iterator    lru_it;
  };

  static size_t clamp(size_t n) {
    return (n > 0 && n < MIN_CACHE_SIZE) ? MIN_CACHE_SIZE : n;
  }

  void evict_lru_locked() {
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] plan_cache: EVICT LRU (size=" << map_.size() - 1 << ")\n";
    auto it = map_.find(*lru_list_.back());
    auto cached = it->second.cached;
    map_.erase(it);
    lru_list_.pop_back();
    auto e1 = cutensorDestroyOperationDescriptor(cached.contractionDesc);
    auto e2 = cutensorDestroyPlan(cached.plan);
    if (e1 != CUTENSOR_STATUS_SUCCESS)
      throw std::runtime_error{std::string{cutensorGetErrorString(e1)}};
    if (e2 != CUTENSOR_STATUS_SUCCESS)
      throw std::runtime_error{std::string{cutensorGetErrorString(e2)}};
  }

  int      log_level_ = 0;
  size_t   max_size_;
  std::unordered_map<ContractionPlanKey, Entry, ContractionPlanKeyHash> map_;
  std::list<const ContractionPlanKey*>                                  lru_list_;
  mutable std::shared_mutex                                             mutex_;
  uint64_t hits_   = 0;
  uint64_t misses_ = 0;
};

BlockSparseContractionPlanCache g_contraction_plan_cache{
    parse_cache_size_env("TAPP_PLAN_CACHE_SIZE", 256,
                         BlockSparseContractionPlanCache::MIN_CACHE_SIZE)};

void wrap_BlockSparseTensorDescriptor(
    cutensorHandle_t& handle,
    const std::vector<int64_t>& numSectionsPerMode,
    const std::vector<int64_t>& sectionExtents,
    const std::vector<int64_t>& blocks,
    const std::vector<int64_t>& strides,
    cudaDataType_t dataType,
    cutensorBlockSparseTensorDescriptor_t& desc
) {
  g_descriptor_cache.get_or_create(
      handle, numSectionsPerMode, sectionExtents, blocks, strides, dataType, desc);
}

} // end anonymous namespace

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

  cutensorHandle_t handle = get_cutensor_handle();

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
  if (tapp_log_level>5) NVTX_MARK("tapp_torch::tensor_product_bs_cuda_impl tensor descriptors");

  /*******************************
   * Block-sparse Contraction.   *
   *******************************/

  // Retrieve (or create) contraction descriptor + plan from cache.
  // NOTE: cuTENSOR API limitation — d_desc is always passed in the C-tensor position.
  //       See https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcreateblocksparsecontractiondescriptor
  ContractionPlanKey plan_key{
      DescriptorKey{a_numSectionsPerMode, a_sectionExtents, a_blocks, a_strides, dtype},
      a_modes_32,
      DescriptorKey{b_numSectionsPerMode, b_sectionExtents, b_blocks, b_strides, dtype},
      b_modes_32,
      DescriptorKey{d_numSectionsPerMode, d_sectionExtents, d_blocks, d_strides, dtype},
      c_modes_32,
      DescriptorKey{d_numSectionsPerMode, d_sectionExtents, d_blocks, d_strides, dtype},
      d_modes_32,
      computeDesc,
      CUTENSOR_WORKSPACE_DEFAULT,
      CUTENSOR_ALGO_DEFAULT,
      CUTENSOR_JIT_MODE_NONE
  };
  cutensorOperationDescriptor_t contractionDesc;
  uint64_t workspaceSizeEstimate;
  cutensorPlan_t plan;
  g_contraction_plan_cache.get_or_create(
      handle, plan_key,
      a_desc, b_desc, d_desc, d_desc,
      contractionDesc, workspaceSizeEstimate, plan);
  if (tapp_log_level>5) NVTX_MARK("tapp_torch::tensor_product_bs_cuda_impl plan cache");

  // See https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcontract 
  // for details on workspace allocation alignment requirements.
  auto workspace = cuda_async_alloc<char>(workspaceSizeEstimate, stream);

  if (tapp_log_level>5) NVTX_MARK( "tapp_torch::cutensorBlockSparseContract start" );
  HANDLE_ERROR(cutensorBlockSparseContract(handle, plan,
              (const void*) &alpha, (const void *const *) A.data(), (const void *const *) B.data(),
              (const void*) &beta,  (const void *const *) (C.has_value() ? C.value().data() : D.data()), 
              (void *const *) D.data(), 
              (void*) workspace.get(), workspaceSizeEstimate, stream));

  if (!stream_ptr) {
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
  }

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
  STD_TORCH_CHECK(a_modes.size()>0 && b_modes.size()>0 && d_modes.size()>0, "Each operand must have at least one mode");

  if (C.defined()) {
    STD_TORCH_CHECK(C.dim() == 1, "Tensor C must be 1D.");
    STD_TORCH_CHECK(A.scalar_type() == C.scalar_type(), "Tensor C must have the same dtype as D (add A and B)");
    STD_TORCH_CHECK(C.device().type() == torch::headeronly::DeviceType::CUDA);
  } else {
    // Check if beta is 0
    // Accept both scalar and rank-1 tensor with a single element
  }
  if (tapp_log_level>5) NVTX_MARK( "tapp_torch::tensor_product_bs_cuda validation" );

  // NOTE https://docs.pytorch.org/cppdocs/stable.html#getting-the-current-cuda-stream
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
    aoti_torch_get_current_cuda_stream(D.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  if (tapp_log_level>5) NVTX_MARK( "tapp_torch::tensor_product_bs_cuda stream" );

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
  if (tapp_log_level>5) NVTX_MARK( "tapp_torch::tensor_product_bs_cuda scalar factors" );

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
    if (tapp_log_level>5) NVTX_MARK( "tapp_torch::tensor_product_bs_cuda device pointers" );

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

// v2: blocks/strides/offsets arrive as CPU int64 Tensors; unpack via bulk memcpy then delegate.
// This avoids the per-element Python→C++ IValue boxing that occurs when passing Sequence[int]
// through the PyTorch custom-op dispatcher for large block arrays.
void tensor_product_bs_v2_cuda(
    const torch::stable::Tensor& A,
    const torch::stable::Tensor& B,
    const torch::stable::Tensor& C,
    torch::stable::Tensor& D,
    const std::vector<int64_t>& a_modes,
    const std::vector<int64_t>& a_numSectionsPerMode,
    const std::vector<int64_t>& a_sectionExtents,
    const torch::stable::Tensor& a_blocks_t,
    const torch::stable::Tensor& a_strides_t,
    const torch::stable::Tensor& a_offsets_t,
    const std::vector<int64_t>& b_modes,
    const std::vector<int64_t>& b_numSectionsPerMode,
    const std::vector<int64_t>& b_sectionExtents,
    const torch::stable::Tensor& b_blocks_t,
    const torch::stable::Tensor& b_strides_t,
    const torch::stable::Tensor& b_offsets_t,
    const std::optional<std::vector<int64_t>>& c_modes,
    const std::optional<std::vector<int64_t>>& c_numSectionsPerMode,
    const std::optional<std::vector<int64_t>>& c_sectionExtents,
    const torch::stable::Tensor& c_blocks_t,
    const torch::stable::Tensor& c_strides_t,
    const torch::stable::Tensor& c_offsets_t,
    const std::vector<int64_t>& d_modes,
    const std::vector<int64_t>& d_numSectionsPerMode,
    const std::vector<int64_t>& d_sectionExtents,
    const torch::stable::Tensor& d_blocks_t,
    const torch::stable::Tensor& d_strides_t,
    const torch::stable::Tensor& d_offsets_t,
    const torch::stable::Tensor& alpha_t,
    const torch::stable::Tensor& beta_t
) {
  const char* env = std::getenv("TAPP_LOG_LEVEL");
  int tapp_log_level = (env) ? std::atoi(env) : 0;

  auto t2v = [](const torch::stable::Tensor& t) -> std::vector<int64_t> {
    const auto* p = static_cast<const int64_t*>(t.const_data_ptr());
    return std::vector<int64_t>(p, p + t.numel());
  };

  std::optional<std::vector<int64_t>> c_blocks, c_strides, c_offsets;
  if (c_blocks_t.defined()) {
    c_blocks  = t2v(c_blocks_t);
    c_strides = t2v(c_strides_t);
    c_offsets = t2v(c_offsets_t);
  }
  if (tapp_log_level>5) NVTX_MARK( "tapp_torch::tensor_product_bs_cuda_v2 t2v" );

  tensor_product_bs_cuda(A, B, C, D,
    a_modes, a_numSectionsPerMode, a_sectionExtents, t2v(a_blocks_t), t2v(a_strides_t), t2v(a_offsets_t),
    b_modes, b_numSectionsPerMode, b_sectionExtents, t2v(b_blocks_t), t2v(b_strides_t), t2v(b_offsets_t),
    c_modes, c_numSectionsPerMode, c_sectionExtents, c_blocks, c_strides, c_offsets,
    d_modes, d_numSectionsPerMode, d_sectionExtents, t2v(d_blocks_t), t2v(d_strides_t), t2v(d_offsets_t),
    alpha_t, beta_t);
}

// Registers CUDA implementation
STABLE_TORCH_LIBRARY_IMPL(tapp_torch, CUDA, m) {
  m.impl("tensor_product_bs",    TORCH_BOX(&tensor_product_bs_cuda));
  m.impl("tensor_product_bs_v2", TORCH_BOX(&tensor_product_bs_v2_cuda));
}

// ── Descriptor-cache management ────────────────────────────────────────────
// These ops carry no tensor arguments so they dispatch via the CPU key.
// The cache itself is process-global; ops are available after _C_cuda is loaded.

void descriptor_cache_clear_impl() {
  g_descriptor_cache.clear();
}

void descriptor_cache_set_max_size_impl(int64_t n) {
  g_descriptor_cache.set_max_size(static_cast<size_t>(n));
}

std::tuple<int64_t, int64_t> descriptor_cache_stats_impl() {
  auto [hits, misses] = g_descriptor_cache.stats();
  return {static_cast<int64_t>(hits), static_cast<int64_t>(misses)};
}

std::tuple<int64_t, int64_t> descriptor_cache_size_impl() {
  auto [count, bytes] = g_descriptor_cache.size_info();
  return {static_cast<int64_t>(count), static_cast<int64_t>(bytes)};
}

void plan_cache_clear_impl() {
  g_contraction_plan_cache.clear();
}

void plan_cache_set_max_size_impl(int64_t n) {
  g_contraction_plan_cache.set_max_size(static_cast<size_t>(n));
}

std::tuple<int64_t, int64_t> plan_cache_stats_impl() {
  auto [hits, misses] = g_contraction_plan_cache.stats();
  return {static_cast<int64_t>(hits), static_cast<int64_t>(misses)};
}

std::tuple<int64_t, int64_t> plan_cache_size_impl() {
  auto [count, bytes] = g_contraction_plan_cache.size_info();
  return {static_cast<int64_t>(count), static_cast<int64_t>(bytes)};
}

} // namespace tapp_torch

// ── Python module init (_C_cuda) ───────────────────────────────────────────
// Cache management functions have no tensor arguments, so torch dispatch cannot
// route to them automatically.  Expose them as plain module-level functions on
// the _C_cuda extension object instead (same pattern as _C / PyInit__C).

static PyObject* py_descriptor_cache_clear(PyObject*, PyObject*) noexcept {
  try {
    tapp_torch::descriptor_cache_clear_impl();
    Py_RETURN_NONE;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* py_descriptor_cache_set_max_size(PyObject*, PyObject* args) noexcept {
  long long n;
  if (!PyArg_ParseTuple(args, "L", &n)) return nullptr;
  try {
    tapp_torch::descriptor_cache_set_max_size_impl(static_cast<int64_t>(n));
    Py_RETURN_NONE;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* py_descriptor_cache_stats(PyObject*, PyObject*) noexcept {
  try {
    auto [hits, misses] = tapp_torch::descriptor_cache_stats_impl();
    return Py_BuildValue("(LL)", static_cast<long long>(hits),
                                 static_cast<long long>(misses));
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* py_descriptor_cache_size(PyObject*, PyObject*) noexcept {
  try {
    auto [count, bytes] = tapp_torch::descriptor_cache_size_impl();
    return Py_BuildValue("(LL)", static_cast<long long>(count),
                                 static_cast<long long>(bytes));
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* py_plan_cache_clear(PyObject*, PyObject*) noexcept {
  try {
    tapp_torch::plan_cache_clear_impl();
    Py_RETURN_NONE;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* py_plan_cache_set_max_size(PyObject*, PyObject* args) noexcept {
  long long n;
  if (!PyArg_ParseTuple(args, "L", &n)) return nullptr;
  try {
    tapp_torch::plan_cache_set_max_size_impl(static_cast<int64_t>(n));
    Py_RETURN_NONE;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* py_plan_cache_stats(PyObject*, PyObject*) noexcept {
  try {
    auto [hits, misses] = tapp_torch::plan_cache_stats_impl();
    return Py_BuildValue("(LL)", static_cast<long long>(hits),
                                 static_cast<long long>(misses));
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* py_plan_cache_size(PyObject*, PyObject*) noexcept {
  try {
    auto [count, bytes] = tapp_torch::plan_cache_size_impl();
    return Py_BuildValue("(LL)", static_cast<long long>(count),
                                 static_cast<long long>(bytes));
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyMethodDef _C_cuda_methods[] = {
  {"descriptor_cache_clear",        py_descriptor_cache_clear,        METH_NOARGS,
   "Clear the descriptor cache and reset hit/miss counters."},
  {"descriptor_cache_set_max_size", py_descriptor_cache_set_max_size, METH_VARARGS,
   "Set the maximum number of cached descriptors. 0 = unlimited."},
  {"descriptor_cache_stats",        py_descriptor_cache_stats,        METH_NOARGS,
   "Return (hits, misses) from the descriptor cache."},
  {"descriptor_cache_size",         py_descriptor_cache_size,         METH_NOARGS,
   "Return (count, host_bytes) from the descriptor cache."},
  {"plan_cache_clear",              py_plan_cache_clear,              METH_NOARGS,
   "Clear the contraction plan cache and reset hit/miss counters."},
  {"plan_cache_set_max_size",       py_plan_cache_set_max_size,       METH_VARARGS,
   "Set the maximum number of cached plans. 0 = unlimited."},
  {"plan_cache_stats",              py_plan_cache_stats,              METH_NOARGS,
   "Return (hits, misses) from the contraction plan cache."},
  {"plan_cache_size",               py_plan_cache_size,               METH_NOARGS,
   "Return (count, host_bytes) from the contraction plan cache."},
  {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef _C_cuda_module_def = {
  PyModuleDef_HEAD_INIT, "_C_cuda", nullptr, -1, _C_cuda_methods
};

extern "C" {
  PyObject* PyInit__C_cuda(void) {
    return PyModule_Create(&_C_cuda_module_def);
  }
}