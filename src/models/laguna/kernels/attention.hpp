#ifndef GUFO_MODELS_LAGUNA_KERNELS_ATTENTION_HPP_
#define GUFO_MODELS_LAGUNA_KERNELS_ATTENTION_HPP_

// GPU attention block for Laguna-S-2.1 (gfx1151), prefill chunks and decode.
// It follows docs/models/laguna-s-2.1/ARCH.md and is checked against the
// phase 2 CPU oracle (`reference::AttnForward`).
//
// On n = RMSNorm(h) * attn_norm (the operator takes n, like FfnOps takes the
// post-ffn_norm rows), for T rows at positions start_pos .. start_pos+T-1:
//
//   q = n Wq, k = n Wk, v = n Wv            Q6_K, Q8_1/MMQ activations
//   g = n Wg                                F32 [heads][hidden], exact
//   q, k    per-head RMSNorm (eps, weight [128]) then NeoX RoPE on the
//           leading rotary_dim channels (YaRN on full layers)
//   k, v    appended to the layer's F16 KV cache
//   a       causal softmax attention, scale 1/sqrt(128), GQA, window on SWA
//   out = (a * softplus(g)) Wo              per head, then Q6_K Wo
//
// Full layers keep every position (capacity = context). SWA layers keep a
// ring of `window + max_chunk` positions: position p lives in slot
// p % capacity, and a chunk of at most max_chunk rows is appended before it is
// attended, which cannot overwrite a key the chunk's first row still sees.
//
// The cache tracks the history it physically retains separately from its
// logical end: committed() is where the next call may continue, and
// retained_begin() the earliest position whose slot still holds that
// position's K/V. Writing position p evicts p - capacity, and a rewind (a call
// below committed()) lowers the end but cannot restore evicted slots, so a
// call is accepted only when every key its rows read before start_pos lies in
// [retained_begin(), committed()). Every argument is validated before any work
// is enqueued; a failure after the call's K/V append was enqueued marks the
// cache invalid (committed() 0, valid() false) and it rejects every call until
// Reset().
//
// The attention kernel is row independent: row p sums its keys in an order
// fixed by p alone, so a row's result does not depend on how the sequence was
// split into chunks. The projections do depend on the Path (see ffn.hpp): the
// vector (MMVQ) and batched (MMQ) kernels round activations differently, so
// compare runs of the same Path.
//
// Scratch is owned by the AttnOps instance and every launch is asynchronous on
// the instance's stream: use one AttnOps (and one KvCache per layer and
// sequence) per stream. Device inputs and outputs must stay valid until the
// stream has run past the call. Growing a scratch buffer allocates the
// replacement first and swaps it in only on success, so a failed allocation
// leaves the instance exactly as it was.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/laguna/config.hpp"
#include "src/models/laguna/kernels/ffn.hpp"
#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna::rocm {

/// Rope constants of one layer (see reference::MakeRopeSetup, which computes
/// them for the oracle too).
struct RopeSpec {
  std::uint32_t rotary_dim{0};  ///< Leading channels rotated, even, <= 128.
  float theta_scale{1.0F};      ///< freq_base^(-2/rotary_dim)
  float corr_low{0.0F};         ///< YaRN ramp start (pair index)
  float corr_high{0.0F};        ///< YaRN ramp end
  float freq_scale{1.0F};       ///< 1/factor with YaRN, else 1
  float ext_factor{0.0F};       ///< 1 with YaRN, else 0
  float mscale{1.0F};           ///< multiplies cos and sin
};

/// Device view of one layer's attention tensors.
struct AttnWeights {
  DeviceMatrix q;                ///< Q6_K [hidden -> heads*128]
  DeviceMatrix k;                ///< Q6_K [hidden -> kv_heads*128]
  DeviceMatrix v;                ///< Q6_K [hidden -> kv_heads*128]
  DeviceMatrix o;                ///< Q6_K [heads*128 -> hidden]
  const float* gate{nullptr};    ///< F32 [heads][hidden]
  const float* q_norm{nullptr};  ///< F32 [128]
  const float* k_norm{nullptr};  ///< F32 [128]
  std::uint32_t hidden{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{128};
  std::uint32_t window{0};  ///< 0: full causal; else key j seen iff p-j<window
  float eps{1e-6F};
  RopeSpec rope;
};

/// Owning device copy of one layer's attention tensors. Move-only.
class LayerAttn {
public:
  LayerAttn() = default;
  LayerAttn(const LayerAttn&) = delete;
  LayerAttn& operator=(const LayerAttn&) = delete;
  LayerAttn(LayerAttn&& other) noexcept;
  LayerAttn& operator=(LayerAttn&& other) noexcept;
  ~LayerAttn();

  /// Copies layer `il`'s attention tensors. Returns nullopt with a message on
  /// a type other than Q6_K (projections) / F32 (gate, norms), a shape that
  /// disagrees with `config`, or a HIP failure (partial uploads are freed).
  [[nodiscard]] static std::optional<LayerAttn> Upload(
      const Config& config, const LayerWeights& layer, std::uint32_t il,
      std::string* error_msg = nullptr);

  [[nodiscard]] const AttnWeights& weights() const noexcept { return w_; }
  [[nodiscard]] std::size_t device_bytes() const noexcept { return bytes_; }

private:
  void Release() noexcept;
  AttnWeights w_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

/// Host copy of the history one KvCache physically retains: the K/V rows of
/// positions [retained_begin, committed) in position order, plus the
/// bookkeeping that says which positions those are. Restoring it into a cache
/// of the same geometry reproduces the cache exactly as far as any later call
/// can observe it (slots outside the range are never read).
struct KvImage {
  std::uint32_t committed{0};
  std::uint32_t retained_begin{0};
  std::vector<std::uint8_t> keys;    ///< (committed - retained_begin) rows, F16
  std::vector<std::uint8_t> values;  ///< same layout
  [[nodiscard]] std::size_t bytes() const noexcept {
    return keys.size() + values.size();
  }
};

/// F16 K/V cache of one layer of one sequence. Move-only.
class KvCache {
public:
  KvCache() = default;
  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;
  KvCache(KvCache&& other) noexcept;
  KvCache& operator=(KvCache&& other) noexcept;
  ~KvCache();

  /// Cache for layer `il`: a full layer holds `context` positions, a sliding
  /// layer a ring of `window + max_chunk` (calls may carry up to max_chunk
  /// rows). Nullopt with a message on a HIP failure.
  [[nodiscard]] static std::optional<KvCache> Create(
      const Config& config, std::uint32_t il, std::uint32_t context,
      std::uint32_t max_chunk, std::string* error_msg = nullptr);
  /// Explicit geometry (window 0: full layer). Row = kv_heads * head_dim.
  [[nodiscard]] static std::optional<KvCache> Create(
      std::uint32_t kv_heads, std::uint32_t head_dim, std::uint32_t capacity,
      std::uint32_t window, std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::uint32_t window() const noexcept { return window_; }
  [[nodiscard]] std::uint32_t kv_heads() const noexcept { return kv_heads_; }
  [[nodiscard]] std::uint32_t head_dim() const noexcept { return head_dim_; }
  /// Positions appended so far (the next call's start_pos at most).
  [[nodiscard]] std::uint32_t committed() const noexcept { return committed_; }
  /// Earliest position whose K/V is still in its slot for the current
  /// history: [retained_begin(), committed()) is physically present. Always 0
  /// on a full layer.
  [[nodiscard]] std::uint32_t retained_begin() const noexcept {
    return retained_begin_;
  }
  /// False after a call failed once it may have written K/V; every Forward is
  /// then rejected until Reset().
  [[nodiscard]] bool valid() const noexcept { return valid_; }
  /// Empties the cache (committed 0, valid). Host state only: the device
  /// buffers are kept and overwritten by later calls, so in-flight work on
  /// them may continue.
  void Reset() noexcept {
    committed_ = 0;
    retained_begin_ = 0;
    valid_ = true;
  }
  /// Bytes of one position's K (or V) row.
  [[nodiscard]] std::size_t row_bytes() const noexcept {
    return std::size_t{kv_heads_} * head_dim_ * 2;
  }
  /// Copies the retained history to the host. `stream` must be the stream the
  /// cache's calls ran on; it is synchronized first. Fails (and fills
  /// nothing) on an invalid cache or a HIP error.
  [[nodiscard]] bool Export(hipStream_t stream, KvImage* image,
                            std::string* error_msg = nullptr) const;
  /// Replaces the cache's content with `image` (any previous content,
  /// including an invalid or failed one, is discarded) and sets committed(),
  /// retained_begin() and valid() accordingly. Fails without touching the
  /// cache when the image does not fit the geometry.
  [[nodiscard]] bool Import(hipStream_t stream, const KvImage& image,
                            std::string* error_msg = nullptr);
  /// Slot of position `p`.
  [[nodiscard]] std::uint32_t Slot(std::uint32_t p) const noexcept {
    return p % capacity_;
  }
  /// [capacity][kv_heads][head_dim] F16 (raw bits), for tests.
  [[nodiscard]] const void* keys() const noexcept { return k_; }
  [[nodiscard]] const void* values() const noexcept { return v_; }
  [[nodiscard]] std::size_t device_bytes() const noexcept;

private:
  friend class AttnOps;
  void Release() noexcept;
  void* k_{nullptr};
  void* v_{nullptr};
  std::uint32_t kv_heads_{0};
  std::uint32_t head_dim_{0};
  std::uint32_t capacity_{0};
  std::uint32_t window_{0};
  std::uint32_t committed_{0};
  std::uint32_t retained_begin_{0};
  bool valid_{true};
};

/// One decode row of a batch, as the kernels read it: the K/V buffers of the
/// row's own sequence (this layer), the position the row is written at and
/// attends from, and the buffers' capacity in positions.
struct DecodeRow {
  void* keys{nullptr};
  void* values{nullptr};
  std::uint32_t pos{0};
  std::uint32_t capacity{0};
};

/// Device allocation seam (tests inject failures here); defaults to
/// hipMalloc / hipFree.
struct DeviceAllocator {
  hipError_t (*allocate)(void** ptr, std::size_t bytes){nullptr};
  hipError_t (*release)(void* ptr){nullptr};
};

class AttnOps {
public:
  /// `stream` must outlive this object. Calls qfn_mmq_init.
  explicit AttnOps(hipStream_t stream);
  AttnOps(const AttnOps&) = delete;
  AttnOps& operator=(const AttnOps&) = delete;
  ~AttnOps();

  /// Test hook: replaces the scratch allocator. Call before the first Forward
  /// (or after the instance has no scratch); never with work in flight.
  void SetAllocator(const DeviceAllocator& allocator) {
    allocator_ = allocator;
  }

  /// out[t][hidden] = attention block of n[t][hidden] at position
  /// start_pos + t, appending the rows' K/V to `cache`. start_pos may not
  /// exceed cache.committed() (a smaller value rewinds and overwrites; on a
  /// ring it may rewind by at most capacity - window, and only while the keys
  /// [start_pos - window + 1, start_pos) are still retained, see
  /// KvCache::retained_begin()). Fails without touching the cache on a bad
  /// argument, including an invalid cache; a failure after the K/V append was
  /// enqueued invalidates the cache (see KvCache::valid()). Rows per call: at
  /// most 32 on Path::kVector; at most capacity - window on a ring;
  /// start_pos + tokens <= capacity on a full layer.
  [[nodiscard]] bool Forward(const AttnWeights& w, KvCache& cache,
                             const float* n, float* out, std::uint32_t tokens,
                             std::uint32_t start_pos, Path path = Path::kAuto,
                             std::string* error_msg = nullptr);

  /// Decode row of `cache` at its committed() position (for ForwardBatch).
  [[nodiscard]] static DecodeRow DescribeDecodeRow(const KvCache& cache);

  /// One decode row for each of several sequences: out[t][hidden] = the
  /// attention block of n[t][hidden] at position caches[t]->committed(),
  /// appending the row's K/V to caches[t]. Each row is computed exactly as
  /// Forward() computes the same row alone on Path::kVector (the projections
  /// share one pass, the kernels differ only in where a row's cache, position
  /// and capacity come from), so a row does not depend on its peers.
  /// `device_rows` is a device table of DescribeDecodeRow(*caches[t]) taken
  /// before the call. At most 32 caches, all distinct. A bad argument touches
  /// nothing; a failure after the K/V append was enqueued invalidates every
  /// cache of the call.
  [[nodiscard]] bool ForwardBatch(const AttnWeights& w,
                                  std::span<KvCache* const> caches,
                                  const DecodeRow* device_rows, const float* n,
                                  float* out, std::string* error_msg = nullptr);

  /// Debug tap for stage-by-stage checks: the operator synchronizes the stream
  /// after each stage and calls `tap` with the stage name, a device pointer and
  /// a float count. Stages: q_proj, k_proj, v_proj, gate_proj (n times the
  /// projections), q_rope (q after norm and RoPE), attn (before the gate),
  /// gate_act (softplus), gated, o (the output). K/V rows of the call are in
  /// the cache after the "attn" stage. Taps hide stream races, so every check
  /// also runs untapped.
  using TapFn = std::function<void(std::string_view name, const float* device,
                                   std::size_t count)>;
  void SetTap(TapFn tap) { tap_ = std::move(tap); }

  /// Test hook: Forward fails right after emitting stage `stage` (one of the
  /// tap stage names; empty disables), whether or not a tap is set. Used to
  /// exercise failures after the K/V append.
  void SetFailAfterStage(std::string stage) { fail_after_ = std::move(stage); }

private:
  /// One device scratch buffer; grows by allocate-then-swap.
  struct Scratch {
    void* ptr{nullptr};
    std::size_t capacity{0};
  };
  bool Run(const AttnWeights& w, std::span<KvCache* const> caches,
           const DecodeRow* rows, const float* n, float* out,
           std::uint32_t tokens, std::uint32_t start_pos, Path path,
           std::string* error_msg);
  bool Reserve(Scratch* buffer, std::size_t bytes, std::string* error_msg);
  void Free(Scratch* buffer) noexcept;
  bool Project(const DeviceMatrix& w, const float* x, float* out,
               std::uint32_t tokens, bool vector, std::string* error_msg);
  bool Emit(const char* name, const float* device, std::size_t count,
            std::string* error_msg);

  hipStream_t stream_;
  DeviceAllocator allocator_;
  TapFn tap_;
  std::string fail_after_;
  bool initialized_{false};
  Scratch q_, k_, v_, gate_, attn_, ids_;
};

}  // namespace gufo::models::laguna::rocm

#endif  // GUFO_MODELS_LAGUNA_KERNELS_ATTENTION_HPP_
