#pragma once

// Block-sparse tensor descriptor + contraction-plan caches used by cutensor_contract_bs.cu.
//
// Split out purely to keep cutensor_contract_bs.cu (the actual contraction implementation)
// navigable; everything here is types/templates/inline functions with no mutable state, so it's
// a plain header rather than a separate translation unit (see cutensor_contract_bs.cu, where
// the process-global cache instances live, for why: the cache code used to sit in that file's
// anonymous namespace, and a real separate .cu/.cpp would need external-linkage symbols plus a
// new build entry for no benefit, since none of this needs to be compiled by nvcc — no
// __global__/kernel code, only host-side C++ and cuTENSOR API calls).
//
// Depends on macros defined by the includer before this header is included:
//   - HANDLE_ERROR(x)     — cutensor_contract_bs.cu, wraps a cutensorStatus_t-returning call.
//   - STD_TORCH_CHECK(...) — from torch/headeronly, used by unpack_descriptor_hashes.

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

namespace tapp_torch {
namespace blocksparse_cache {

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

// A 512-bit digest, supplied by the caller as a substitute for hashing/comparing the
// (potentially very large) numSectionsPerMode+sectionExtents+blocks+strides arrays that make up
// a DescriptorKey. Callers guarantee the digest is unique to that combination, so it can serve as
// a full identity for cache lookups: no fallback comparison against the source arrays is done.
struct Hash512 {
  uint64_t w[8];
  bool operator==(const Hash512& o) const {
    for (int i = 0; i < 8; ++i) if (w[i] != o.w[i]) return false;
    return true;
  }
};

// Unpacks a flat int64[24] (3 consecutive 512-bit hashes) into per-tensor digests for A, B, D.
inline std::array<Hash512, 3> unpack_descriptor_hashes(const std::vector<int64_t>& flat) {
  STD_TORCH_CHECK(flat.size() == 24,
      "descriptor_key_hashes must have length 24 (3 hashes x 8 int64 each, for A, B, D)");
  std::array<Hash512, 3> result;
  for (int h = 0; h < 3; ++h)
    for (int i = 0; i < 8; ++i)
      result[h].w[i] = static_cast<uint64_t>(flat[h * 8 + i]);
  return result;
}

// Lightweight cache key used when the caller supplies a precomputed Hash512 instead of the raw
// descriptor-defining arrays. dataType is folded in separately since it is not part of the
// caller-supplied digest.
struct HashedDescriptorKey {
  Hash512 hash;
  cudaDataType_t dataType;
  bool operator==(const HashedDescriptorKey& o) const {
    return hash == o.hash && dataType == o.dataType;
  }
};

struct HashedDescriptorKeyHash {
  size_t operator()(const HashedDescriptorKey& k) const {
    // The caller-supplied hash is already high-entropy, so a cheap O(1) fold is sufficient
    // (as opposed to DescriptorKeyHash, which must scan the O(nBlocks) source arrays).
    size_t s = static_cast<size_t>(k.hash.w[0]);
    for (int i = 1; i < 8; ++i)
      s ^= static_cast<size_t>(k.hash.w[i]) + 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= std::hash<int>{}(static_cast<int>(k.dataType)) + 0x9e3779b9 + (s << 6) + (s >> 2);
    return s;
  }
};

inline size_t key_storage_bytes(const DescriptorKey& k) {
  return sizeof(int64_t) * (k.numSectionsPerMode.size() + k.sectionExtents.size()
                           + k.blocks.size() + k.strides.size())
       + sizeof(cudaDataType_t);
}

inline size_t key_storage_bytes(const HashedDescriptorKey&) {
  return sizeof(Hash512) + sizeof(cudaDataType_t);
}

// Descriptors encode only tensor metadata and are safe to reuse across handle instances.
// Ownership stays in the cache; callers must NOT call cutensorDestroyBlockSparseTensorDescriptor.
// max_size=0 means unlimited. Eviction policy: LRU.
// A single contraction needs up to 4 descriptors (A, B, C, D), so the minimum
// enforced capacity is MIN_CACHE_SIZE when a finite limit is requested.
// Templated on the key type so the same cache implementation serves both the
// vector-based DescriptorKey (hash computed internally) and the lightweight
// HashedDescriptorKey (hash supplied by the caller).
template <typename KeyT, typename KeyHashT>
class DescriptorCacheImpl {
public:
  static constexpr size_t MIN_CACHE_SIZE = 4;

  explicit DescriptorCacheImpl(size_t max_size)
      : max_size_(clamp_cache_size(max_size)) {
    const char* env = std::getenv("TAPP_LOG_LEVEL");
    log_level_ = env ? std::atoi(env) : 0;
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] desc_cache: created max_size="
                << (max_size_ == 0 ? "unlimited" : std::to_string(max_size_)) << "\n";
  }

  void get_or_create(
      cutensorHandle_t& handle,
      const KeyT& key,
      const std::vector<int64_t>& numSectionsPerMode,
      const std::vector<int64_t>& sectionExtents,
      const std::vector<int64_t>& blocks,
      const std::vector<int64_t>& strides,
      cudaDataType_t dataType,
      cutensorBlockSparseTensorDescriptor_t& desc
  ) {
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
    auto map_it = map_.emplace(key, Entry{desc, lru_list_.end()}).first;
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
      bytes += key_storage_bytes(key) + sizeof(cutensorBlockSparseTensorDescriptor_t);
    return {map_.size(), bytes};
  }

  ~DescriptorCacheImpl() noexcept(false) {
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
    cutensorBlockSparseTensorDescriptor_t desc;
    typename std::list<const KeyT*>::iterator lru_it;
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
  std::unordered_map<KeyT, Entry, KeyHashT> map_;
  std::list<const KeyT*>                    lru_list_;
  mutable std::shared_mutex                 mutex_;
  uint64_t hits_   = 0;
  uint64_t misses_ = 0;
};

using BlockSparseDescriptorCache = DescriptorCacheImpl<DescriptorKey, DescriptorKeyHash>;
using HashedDescriptorCache      = DescriptorCacheImpl<HashedDescriptorKey, HashedDescriptorKeyHash>;

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

// ── Contraction descriptor + plan cache ──────────────────────────────────
// Key covers all inputs to cutensorCreateBlockSparseContraction +
// cutensorEstimateWorkspaceSize.  The c position always receives d_desc due
// to a cuTENSOR API limitation, so c_key is always the D descriptor key.
// Templated on the descriptor-key type for the same reason as DescriptorCacheImpl above.
template <typename DKey>
struct ContractionPlanKeyT {
  DKey                  a_key;  std::vector<int32_t> a_modes;
  DKey                  b_key;  std::vector<int32_t> b_modes;
  DKey                  c_key;  std::vector<int32_t> c_modes;
  DKey                  d_key;  std::vector<int32_t> d_modes;
  cutensorComputeDescriptor_t  computeDesc;
  cutensorWorksizePreference_t workspacePref;
  cutensorAlgo_t               algo;
  cutensorJitMode_t            jitMode;

  bool operator==(const ContractionPlanKeyT& o) const {
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

template <typename DKey, typename DKeyHash>
struct ContractionPlanKeyHashT {
  size_t operator()(const ContractionPlanKeyT<DKey>& k) const {
    auto combine = [](size_t seed, size_t h) -> size_t {
      return seed ^ (h + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    };
    auto hash_modes = [&](const std::vector<int32_t>& v) -> size_t {
      size_t h = v.size();
      for (auto x : v) h = combine(h, std::hash<int32_t>{}(x));
      return h;
    };
    DKeyHash dkh;
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

using ContractionPlanKey           = ContractionPlanKeyT<DescriptorKey>;
using ContractionPlanKeyHash       = ContractionPlanKeyHashT<DescriptorKey, DescriptorKeyHash>;
using HashedContractionPlanKey     = ContractionPlanKeyT<HashedDescriptorKey>;
using HashedContractionPlanKeyHash = ContractionPlanKeyHashT<HashedDescriptorKey, HashedDescriptorKeyHash>;

template <typename PlanKeyT, typename PlanKeyHashT>
class BlockSparseContractionPlanCacheImpl {
public:
  static constexpr size_t MIN_CACHE_SIZE = 1;

  explicit BlockSparseContractionPlanCacheImpl(size_t max_size)
      : max_size_(clamp(max_size)) {
    const char* env = std::getenv("TAPP_LOG_LEVEL");
    log_level_ = env ? std::atoi(env) : 0;
    if (log_level_ >= 6)
      std::cout << "[tapp_torch] plan_cache: created max_size="
                << (max_size_ == 0 ? "unlimited" : std::to_string(max_size_)) << "\n";
  }

  void get_or_create(
      cutensorHandle_t&                     handle,
      const PlanKeyT&                       key,
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

    auto map_it = map_.emplace(key,
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
      auto modes_bytes = [](const std::vector<int32_t>& v) {
        return sizeof(int32_t) * v.size();
      };
      bytes += key_storage_bytes(key.a_key) + modes_bytes(key.a_modes)
             + key_storage_bytes(key.b_key) + modes_bytes(key.b_modes)
             + key_storage_bytes(key.c_key) + modes_bytes(key.c_modes)
             + key_storage_bytes(key.d_key) + modes_bytes(key.d_modes)
             + sizeof(cutensorComputeDescriptor_t) + sizeof(cutensorWorksizePreference_t)
             + sizeof(cutensorOperationDescriptor_t) + sizeof(cutensorPlan_t);
    }
    return {map_.size(), bytes};
  }

  ~BlockSparseContractionPlanCacheImpl() noexcept(false) {
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
    Cached                                       cached;
    typename std::list<const PlanKeyT*>::iterator lru_it;
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
  std::unordered_map<PlanKeyT, Entry, PlanKeyHashT> map_;
  std::list<const PlanKeyT*>                        lru_list_;
  mutable std::shared_mutex                         mutex_;
  uint64_t hits_   = 0;
  uint64_t misses_ = 0;
};

using BlockSparseContractionPlanCache =
    BlockSparseContractionPlanCacheImpl<ContractionPlanKey, ContractionPlanKeyHash>;
using HashedBlockSparseContractionPlanCache =
    BlockSparseContractionPlanCacheImpl<HashedContractionPlanKey, HashedContractionPlanKeyHash>;

} // namespace blocksparse_cache
} // namespace tapp_torch
