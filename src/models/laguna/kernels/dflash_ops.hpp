#ifndef GUFO_MODELS_LAGUNA_KERNELS_DFLASH_OPS_HPP_
#define GUFO_MODELS_LAGUNA_KERNELS_DFLASH_OPS_HPP_

// GPU operators of the Laguna DFlash drafter (gfx1151), phase 6c of
// docs/models/laguna-s-2.1/DFLASH.md. Each is checked against the phase 6b CPU
// oracle (`reference::DFlashModel`, src/models/laguna/dflash_reference.*).
//
// Encoder, per context row t (no dependence between rows):
//
//   a_k   = RMSNorm(feat[t][k*H .. (k+1)*H)) * aux_norm[k]     k < taps
//   u     = fc . [a_0 | ... | a_{taps-1}]                      Q4_K
//   fused = RMSNorm(u) * enc_output_norm
//
// The projections reuse the qfn_mmq kernels vendored with qwen38_flash_next
// with one expert and zero ids (the target's head does the same): the vector
// (MMVQ) path quantizes activations to Q8_1, the batched (MMQ tile) path to the
// MMQ layout, so the two paths round differently; compare runs of one Path.
//
// Scratch is owned by the DFlashOps instance and every launch is asynchronous
// on the instance's stream: use one DFlashOps per stream. Growing a scratch
// buffer allocates the replacement first and swaps it in only on success, so a
// failed allocation leaves the instance exactly as it was.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/laguna/dflash.hpp"
#include "src/models/laguna/kernels/attention.hpp"
#include "src/models/laguna/kernels/ffn.hpp"
#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna::rocm {

/// Device view of one drafter layer. Projections are Q4_K or Q6_K.
struct DrafterLayer {
  const float* attn_norm{nullptr};  ///< F32 [hidden]
  DeviceMatrix q;                   ///< [hidden -> heads*128]
  DeviceMatrix k;                   ///< [hidden -> kv_heads*128]
  DeviceMatrix v;                   ///< [hidden -> kv_heads*128]
  DeviceMatrix o;                   ///< [heads*128 -> hidden]
  DeviceMatrix gate;                ///< [hidden -> heads], softplus gate
  const float* q_norm{nullptr};     ///< F32 [128]
  const float* k_norm{nullptr};     ///< F32 [128]
  const float* ffn_norm{nullptr};   ///< F32 [hidden]
  DeviceMatrix ffn_gate;            ///< [hidden -> ffn]
  DeviceMatrix ffn_up;              ///< [hidden -> ffn]
  DeviceMatrix ffn_down;            ///< [ffn -> hidden]
};

/// Device view of the whole drafter. Plain data: tests may copy it and change
/// a field to check that the checks notice (wrong theta, window, swapped K/V).
struct DrafterWeights {
  std::uint32_t hidden{0};
  std::uint32_t ffn{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{128};
  std::uint32_t taps{0};
  /// Every layer is sliding: key j is visible from p iff p - j < window.
  std::uint32_t window{0};
  std::uint32_t block_size{0};  ///< Rows per block including the anchor.
  float eps{1e-6F};
  RopeSpec rope;  ///< NeoX, no YaRN (ext_factor 0, mscale 1).
  /// Block rows see block rows 0..r only (Laguna's decoder). False makes the
  /// block non-causal (generic DFlash); kept as data so a test can flip it.
  bool causal_block{true};
  const float* aux_norm{nullptr};     ///< F32 [taps][hidden]
  DeviceMatrix fc;                    ///< [taps*hidden -> hidden]
  const float* enc_norm{nullptr};     ///< F32 [hidden]
  const float* output_norm{nullptr};  ///< F32 [hidden]
  std::vector<DrafterLayer> layers;
};

/// Owning device copy of a drafter (DFlashWeights). Move-only.
class DeviceDrafter {
public:
  DeviceDrafter() = default;
  DeviceDrafter(const DeviceDrafter&) = delete;
  DeviceDrafter& operator=(const DeviceDrafter&) = delete;
  DeviceDrafter(DeviceDrafter&& other) noexcept;
  DeviceDrafter& operator=(DeviceDrafter&& other) noexcept;
  ~DeviceDrafter();

  /// Copies every tensor from TensorRef::data (the mapping or a fixture).
  /// Nullopt with a message on a type other than Q4_K/Q6_K (projections) or
  /// F32 (norms), a shape that disagrees with the config, head_dim != 128, or
  /// a HIP failure (everything allocated is freed).
  [[nodiscard]] static std::optional<DeviceDrafter> Upload(
      const DFlashWeights& weights, std::string* error_msg = nullptr);

  [[nodiscard]] const DrafterWeights& weights() const noexcept { return w_; }
  [[nodiscard]] std::size_t device_bytes() const noexcept { return bytes_; }

private:
  void Release() noexcept;
  DrafterWeights w_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

/// The drafter's K/V of one session: per layer an F16 ring of
/// `window + max(max_rows, block_size)` positions (position p in slot
/// p % capacity). It holds the injected context rows; a block pass writes its
/// own rows at committed() .. committed() + B - 1 and leaves committed()
/// unchanged, so the next injection overwrites them (the fork's
/// seq_rm after the draft pass). Bookkeeping follows the target's KvCache:
/// [retained_begin(), committed()) is physically present, writing position p
/// evicts p - capacity, and a failure after a write invalidates the ring until
/// Reset(). floor() is the first position the session injected (Reset's
/// argument): keys below it never existed and are not attended, as in the
/// fork's cache, which holds only injected rows. Move-only.
class DrafterKv {
public:
  DrafterKv() = default;
  DrafterKv(const DrafterKv&) = delete;
  DrafterKv& operator=(const DrafterKv&) = delete;
  DrafterKv(DrafterKv&& other) noexcept;
  DrafterKv& operator=(DrafterKv&& other) noexcept;
  ~DrafterKv();

  /// Rings for every layer of `w`; injections may carry up to `max_rows`
  /// rows per call. Nullopt with a message on a HIP failure.
  [[nodiscard]] static std::optional<DrafterKv> Create(
      const DrafterWeights& w, std::uint32_t max_rows,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::uint32_t window() const noexcept { return window_; }
  [[nodiscard]] std::uint32_t layers() const noexcept {
    return static_cast<std::uint32_t>(k_.size());
  }
  [[nodiscard]] std::uint32_t kv_heads() const noexcept { return kv_heads_; }
  /// Rows one call may write: capacity - window.
  [[nodiscard]] std::uint32_t max_rows() const noexcept {
    return capacity_ - window_;
  }
  [[nodiscard]] std::uint32_t committed() const noexcept { return committed_; }
  [[nodiscard]] std::uint32_t retained_begin() const noexcept {
    return retained_begin_;
  }
  [[nodiscard]] std::uint32_t floor() const noexcept { return floor_; }
  [[nodiscard]] bool valid() const noexcept { return valid_; }
  /// Empties the rings; the next injection starts at `position`. Host state
  /// only.
  void Reset(std::uint32_t position = 0) noexcept {
    committed_ = position;
    retained_begin_ = position;
    floor_ = position;
    valid_ = true;
  }
  [[nodiscard]] std::uint32_t Slot(std::uint32_t p) const noexcept {
    return p % capacity_;
  }
  /// Bytes of one position's K (or V) row of one layer.
  [[nodiscard]] std::size_t row_bytes() const noexcept {
    return std::size_t{kv_heads_} * 128 * 2;
  }
  /// [capacity][kv_heads][128] F16 (raw bits) of layer `il`, for tests.
  [[nodiscard]] const void* keys(std::uint32_t il) const { return k_.at(il); }
  [[nodiscard]] const void* values(std::uint32_t il) const { return v_.at(il); }
  [[nodiscard]] std::size_t device_bytes() const noexcept {
    return 2 * k_.size() * capacity_ * row_bytes();
  }

private:
  friend class DFlashOps;
  void Release() noexcept;
  std::vector<void*> k_;
  std::vector<void*> v_;
  std::uint32_t kv_heads_{0};
  std::uint32_t capacity_{0};
  std::uint32_t window_{0};
  std::uint32_t committed_{0};
  std::uint32_t retained_begin_{0};
  std::uint32_t floor_{0};
  bool valid_{true};
};

class DFlashOps {
public:
  /// `stream` must outlive this object. Calls qfn_mmq_init.
  explicit DFlashOps(hipStream_t stream);
  DFlashOps(const DFlashOps&) = delete;
  DFlashOps& operator=(const DFlashOps&) = delete;
  ~DFlashOps();

  /// Test hook: replaces the scratch allocator. Never with work in flight.
  void SetAllocator(const DeviceAllocator& allocator) {
    allocator_ = allocator;
  }

  /// fused[t][hidden] = encoder of features[t][taps*hidden] (the target taps
  /// concatenated in DFlashConfig::target_layers order). Rows: at most 32 on
  /// Path::kVector.
  [[nodiscard]] bool Encode(const DrafterWeights& w, const float* features,
                            float* fused, std::uint32_t rows,
                            Path path = Path::kAuto,
                            std::string* error_msg = nullptr);

  /// K/V injection: for every layer l, x = RMSNorm(fused[t]) * attn_norm_l,
  /// K = RoPE(RMSNorm_head(x Wk) * k_norm) at position start + t, V = x Wv,
  /// stored F16 in the layer's ring. start must lie in
  /// [kv.floor(), kv.committed()] (a start below committed() overwrites, as
  /// trimming rejected rows does) and rows <= kv.max_rows(). Fails without
  /// touching the ring on a bad argument, including an invalid ring; a
  /// failure after the first store invalidates it.
  [[nodiscard]] bool InjectKv(const DrafterWeights& w, DrafterKv& kv,
                              const float* fused, std::uint32_t rows,
                              std::uint32_t start, Path path = Path::kAuto,
                              std::string* error_msg = nullptr);

  /// Encode then InjectKv: features[t][taps*hidden] of the context rows at
  /// positions start .. start + rows - 1.
  [[nodiscard]] bool Inject(const DrafterWeights& w, DrafterKv& kv,
                            const float* features, std::uint32_t rows,
                            std::uint32_t start, Path path = Path::kAuto,
                            std::string* error_msg = nullptr);

  /// Attention block of drafter layer `layer` for B block rows at positions
  /// n .. n + B - 1, n = kv.committed(), on their post-attn_norm rows
  /// `normed` ([B][hidden]):
  ///
  ///   q, k, v, g = normed Wq, Wk, Wv, Wg        Q4_K/Q6_K, zero-id qfn_mmq
  ///   q, k       per-head RMSNorm, NeoX RoPE at the row's position
  ///   k, v       F16 into the ring at the block positions
  ///   a          softmax(q K^T / sqrt(128)) V, GQA; row r at p = n + r sees
  ///              keys j with max(p + 1 - window, kv.floor()) <= j <= p
  ///              (the 511 - r context rows and block rows 0..r) or, with
  ///              causal_block false, up to n + B - 1
  ///   out        (a * softplus(g) per head) Wo
  ///
  /// committed() is unchanged (the block rows are overwritten by the next
  /// injection); retained_begin() records what the block rows evicted. B is
  /// at most kv.max_rows() (32 on Path::kVector), and the context keys the
  /// rows read must still be retained. Fails without touching the ring on a
  /// bad argument; a failure after the K/V write invalidates it.
  [[nodiscard]] bool Attention(const DrafterWeights& w, std::uint32_t layer,
                               DrafterKv& kv, const float* normed,
                               std::uint32_t rows, float* out,
                               Path path = Path::kAuto,
                               std::string* error_msg = nullptr);

  /// SwiGLU FFN of drafter layer `layer` on post-ffn_norm rows:
  /// out = (silu(normed Wg) * (normed Wu)) Wd, silu(g) = g / (1 + exp(-g)).
  /// Q4_K gate/up, Q4_K or Q6_K down.
  [[nodiscard]] bool Ffn(const DrafterWeights& w, std::uint32_t layer,
                         const float* normed, float* out, std::uint32_t rows,
                         Path path = Path::kAuto,
                         std::string* error_msg = nullptr);

  /// The drafter's output: result_norm = RMSNorm(h) * output_norm for the B
  /// rows h[B][hidden], logits[B - 1][vocab] = result_norm rows 1..B-1 times
  /// the target's head (`head`, Q4_K/Q6_K [hidden -> vocab]), and drafts[r] =
  /// argmax of logits row r (the first maximum; -1 if the row has no finite
  /// maximum). `logits` and `drafts` are device buffers. B >= 2.
  [[nodiscard]] bool Head(const DrafterWeights& w, const DeviceMatrix& head,
                          const float* h, std::uint32_t rows, float* logits,
                          std::int32_t* drafts, Path path = Path::kAuto,
                          std::string* error_msg = nullptr);

  /// The block forward: B rows [anchor, MASK, ...] with embeddings
  /// `embeddings` ([B][hidden], the target's token_embd rows, no scale) at
  /// positions n .. n + B - 1, n = kv.committed(), through every layer
  /// (attn_norm, Attention, residual, ffn_norm, Ffn, residual), then Head.
  /// 2 <= B <= block_size. Each MASK row predicts the token at its own
  /// position: logits row r - 1 and drafts[r - 1] belong to block row r, the
  /// draft for position n + r.
  /// Fails without touching the ring on a bad argument; a failure after a
  /// K/V write invalidates it.
  [[nodiscard]] bool Block(const DrafterWeights& w, const DeviceMatrix& head,
                           DrafterKv& kv, const float* embeddings,
                           std::uint32_t rows, float* logits,
                           std::int32_t* drafts, Path path = Path::kAuto,
                           std::string* error_msg = nullptr);

  /// Debug tap for stage-by-stage checks: the operator synchronizes the stream
  /// after each stage and calls `tap` with the stage name (the oracle's tap
  /// names, see DFlashModel::Forward, plus the raw projections), a device
  /// pointer and a float count. Taps hide stream races, so every check also
  /// runs untapped. Encoder stages: enc_aux_norm, fc_out, enc_norm_out.
  /// Injection stages per layer N: kv_inp_normed-N, Kproj_injected-N (x Wk),
  /// Vcur_injected-N (x Wv), then kv_stored-N (null pointer, count 0: the
  /// rows are in the ring). Attention stages per layer N: Qproj-N, Kproj-N,
  /// Vcur-N, attn_gate_proj-N (projections), Qcur-N (q after norm and rope;
  /// the block's K/V are in the ring from here), attn_out-N (before the gate),
  /// attn_gate_softplus-N, attn_gated-N, attn_o_proj-N. FFN stages per
  /// layer: ffn_gate-N, ffn_up-N, ffn_swiglu-N, ffn_out-N. Head: result_norm
  /// (all B rows), result_output (rows 1..B-1). Block adds inp_noise_embd,
  /// noise_norm-N, ffn_inp-N, ffn_norm-N and l_out-N.
  using TapFn = std::function<void(std::string_view name, const float* device,
                                   std::size_t count)>;
  void SetTap(TapFn tap) { tap_ = std::move(tap); }

  /// Test hook: the call fails right after stage `stage` (a tap name with its
  /// layer suffix; empty disables), whether or not a tap is set.
  void SetFailAfterStage(std::string stage) { fail_after_ = std::move(stage); }

  /// Bytes of scratch currently held.
  [[nodiscard]] std::size_t scratch_bytes() const noexcept;

private:
  struct Scratch {
    void* ptr{nullptr};
    std::size_t capacity{0};
  };
  bool Reserve(Scratch* buffer, std::size_t bytes, std::string* error_msg);
  void Free(Scratch* buffer) noexcept;
  bool Project(const DeviceMatrix& w, const float* x, float* out,
               std::uint32_t rows, bool vector, std::string* error_msg);
  bool Emit(const char* name, int layer, const float* device, std::size_t count,
            std::string* error_msg);
  bool Ready(Path path, std::uint32_t rows, bool* vector,
             std::string* error_msg) const;
  /// Reserves `rows` zero expert ids (the projections' routing).
  bool PrepareIds(std::uint32_t rows, std::string* error_msg);
  bool EncodeImpl(const DrafterWeights& w, const float* features, float* fused,
                  std::uint32_t rows, bool vector, std::string* error_msg);
  bool InjectImpl(const DrafterWeights& w, DrafterKv& kv, const float* fused,
                  std::uint32_t rows, std::uint32_t start, bool vector,
                  std::string* error_msg);
  bool CheckAttention(const DrafterWeights& w, std::uint32_t layer,
                      const DrafterKv& kv, std::uint32_t rows,
                      std::string* error_msg) const;
  bool AttentionImpl(const DrafterWeights& w, std::uint32_t layer,
                     DrafterKv& kv, const float* normed, std::uint32_t rows,
                     float* out, bool vector, std::string* error_msg);
  bool FfnImpl(const DrafterWeights& w, std::uint32_t layer,
               const float* normed, float* out, std::uint32_t rows, bool vector,
               std::string* error_msg);
  bool HeadImpl(const DrafterWeights& w, const DeviceMatrix& head,
                const float* h, std::uint32_t rows, float* logits,
                std::int32_t* drafts, bool vector, std::string* error_msg);
  bool CheckInject(const DrafterWeights& w, const DrafterKv& kv,
                   std::uint32_t rows, std::uint32_t start,
                   std::string* error_msg) const;

  hipStream_t stream_;
  DeviceAllocator allocator_;
  TapFn tap_;
  std::string fail_after_;
  bool initialized_{false};
  Scratch ids_, aux_, u_, fused_, x_, k_, v_, q_, gate_, attn_;
  Scratch fg_, fu_, fa_, h_, n_, o_, r_;
};

}  // namespace gufo::models::laguna::rocm

#endif  // GUFO_MODELS_LAGUNA_KERNELS_DFLASH_OPS_HPP_
