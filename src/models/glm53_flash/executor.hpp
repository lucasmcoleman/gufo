#ifndef GUFO_MODELS_GLM53_FLASH_EXECUTOR_HPP_
#define GUFO_MODELS_GLM53_FLASH_EXECUTOR_HPP_

// GPU executor for GLM-5.3-Flash (Lane A): the decoder layer loop with
// hyper-connections, the KDA mixer, embedding and head, around Lane B's
// MlaForward / FfnForward. Pieces, bottom up:
//   DeviceArena     owns every device allocation (weights, state, scratch);
//   UploadMatrix    copies one GGUF tensor to the device in its encoding;
//   UploadKdaLayer  fills a KdaLayerDevice; UploadMlaLayer / UploadFfnLayer
//                   fill MlaLayerDevice / FfnLayerDevice (layer_ops.hpp);
//   KdaMixerForward is the KDA counterpart of MlaForward, with the same
//                   OpContext and call conventions (layer_ops.hpp);
//   LayerOps        the injectable MLA / FFN operator table;
//   Executor        uploads a layer range plus embedding / head and runs
//                   Forward (prefill, then incremental decode) on one
//                   sequence.
// The scalar contract is ReferenceModel (reference.cpp) on top of
// reference_{hc,kda}.hpp; the kernels are kernels/{hc,kda}.hpp, gufo's
// batched embedding lookup (any block format, Q6_K included) and RMSNorm.
//
// Projections run through gufo's GEMMs with F32 activations: Q8_0, Q8_K and
// the in-kernel K-quant/IQ formats (Q3_K..Q6_K, IQ4_NL, IQ4_XS, IQ3_S) via
// LaunchBatchedQuantGEMMFp32, F32 and BF16 via LaunchGEMV per token. Other
// encodings are refused at upload. Every KDA projection of the Maya-S GGUF is
// Q8_0 or Q6_K, and so are the embedding and output.weight (Q6_K). MLA and
// FFN weights are uploaded in their GGUF encoding without a route check
// (routed experts are IQ2_XXS etc.); the operators own their formats.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/layer_ops.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace gufo::models::glm53_flash {

/// Owns device allocations; frees them on destruction. Not thread-safe.
class DeviceArena {
public:
  DeviceArena() = default;
  ~DeviceArena();
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  DeviceArena(DeviceArena&& other) noexcept;
  DeviceArena& operator=(DeviceArena&& other) noexcept;

  /// A new uninitialised allocation of `bytes` (256-byte aligned), or nullptr
  /// if the device allocation fails.
  [[nodiscard]] void* Allocate(std::size_t bytes);
  /// Bytes allocated so far.
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  std::vector<void*> blocks_;
  std::size_t bytes_{0};
};

/// True if `type` with `cols` reduction elements has a projection route
/// (see the file comment and the block size of the format).
[[nodiscard]] bool SupportsProjection(core::GgmlType type,
                                      std::uint64_t cols) noexcept;

/// Copies `t` (all experts) to the device unchanged and describes it in
/// `out`. Synchronous. Fails on an empty tensor, an encoding without a
/// projection route, or an allocation / copy failure.
[[nodiscard]] bool UploadMatrix(DeviceArena& arena, const TensorRef& t,
                                DeviceMatrix* out, std::string* error);

/// Y[tokens x w.rows] = X[tokens x w.cols] * W^T for a single-expert matrix,
/// enqueued on `stream`. False on a matrix without a projection route.
[[nodiscard]] bool LaunchProjection(const DeviceMatrix& w, const float* x,
                                    float* y, std::uint32_t tokens,
                                    Stream stream);

/// Device weights of one KDA layer. Projections keep their GGUF encoding;
/// q|k|v are stacked into one [3*kda_dim x hidden] matrix (the conv channel
/// order), which needs a common encoding. F32 vectors are plain floats.
struct KdaLayerDevice {
  DeviceMatrix qkv;       ///< [hidden -> 3*kda_dim], attn_q|attn_k|attn_v
  DeviceMatrix f_a, f_b;  ///< forget gate [hidden -> D], [D -> kda_dim]
  DeviceMatrix g_a, g_b;  ///< output gate [hidden -> D], [D -> kda_dim]
  DeviceMatrix beta;      ///< [hidden -> heads]
  DeviceMatrix out;       ///< attn_output [kda_dim -> hidden]
  const float* conv_w{nullptr};   ///< [3*kda_dim][4], ssm_conv1d_q|k|v
  const float* dt_bias{nullptr};  ///< [kda_dim] ssm_dt.bias
  const float* a{nullptr};        ///< [heads] ssm_a = -exp(A_log)
  const float* norm{nullptr};     ///< [D] ssm_norm, shared by the heads
};

/// Per-layer KDA state of one sequence: conv history and recurrent state.
/// A zero state is a fresh sequence.
struct KdaState {
  float* conv{nullptr};       ///< [3][3*kda_dim] last three q|k|v inputs
  float* recurrent{nullptr};  ///< [heads][D value][D key] (kernels/kda.hpp)
};

/// Uploads layer `layer`'s KDA weights. Fails (with `error`) on a non-KDA
/// layer, a shape that does not match `config`, a head dim other than 128,
/// a conv kernel other than 4, q/k/v in different encodings, an encoding
/// without a projection route, or an allocation failure. Synchronous.
[[nodiscard]] bool UploadKdaLayer(DeviceArena& arena, const Config& config,
                                  const LayerWeights& layer, std::uint32_t il,
                                  KdaLayerDevice* out, std::string* error);

/// Bytes of one layer's KdaState.
[[nodiscard]] std::size_t KdaStateBytes(const Config& config) noexcept;

/// Allocates one layer's KdaState from `arena` and zeroes it (synchronous).
[[nodiscard]] bool AllocateKdaState(DeviceArena& arena, const Config& config,
                                    KdaState* out);

/// Zeroes `state` on `stream` (start of a new sequence).
[[nodiscard]] bool ResetKdaState(const Config& config, const KdaState& state,
                                 Stream stream);

/// Scratch KdaMixerForward needs to run `max_tokens` tokens in one pass.
/// Calls with more tokens than the scratch holds run in chunks, which is
/// exact (the state carries across chunks).
[[nodiscard]] std::size_t KdaScratchBytes(const Config& config,
                                          std::uint32_t max_tokens) noexcept;

/// KDA mixer (Glm5NextTextLinearAttention.forward) on `tokens` consecutive
/// positions of one sequence, continuing from `state` and advancing it:
/// prefill is one call, decode a call per token. `x` is the RMS-normalised
/// mixer input [tokens x hidden], `y` the output [tokens x hidden], both
/// device F32; they may not alias each other or the scratch. Uses
/// ctx.scratch (at least KdaScratchBytes(config, 1)); with ctx.tap set,
/// reports `y` as "kda_out" (llama.cpp's name). Returns false without
/// touching `state` on a bad argument (no KDA layer, null pointer, scratch
/// too small); false after a failed launch leaves `state` unspecified.
[[nodiscard]] bool KdaMixerForward(const OpContext& ctx, const Config& config,
                                   std::uint32_t layer, const KdaLayerDevice& w,
                                   KdaState& state, const float* x,
                                   std::uint32_t tokens, float* y);

// ---- MLA / FFN weights (Lane B's inputs) ----

/// Uploads layer `il`'s MLA + DSA indexer weights into `out` as layer_ops.hpp
/// lays them out: every matrix in its GGUF encoding (k_b / v_b keep their
/// per-head stacks as `experts`), norms, the indexer LayerNorm and the ape
/// table [kpool x indexer_dim] as F32. Fails on a non-MLA layer, a shape
/// that does not match `config`, or an allocation failure. Synchronous.
[[nodiscard]] bool UploadMlaLayer(DeviceArena& arena, const Config& config,
                                  const LayerWeights& layer, std::uint32_t il,
                                  MlaLayerDevice* out, std::string* error);

/// Uploads layer `il`'s FFN: the dense SwiGLU on the leading dense layers,
/// else router (F32), selection bias (F32), the routed expert stacks and the
/// shared expert. Clamps come from config.swiglu_clamp_{exp,shexp}[il] (0
/// past the end). Matrices keep their GGUF encoding. Synchronous.
[[nodiscard]] bool UploadFfnLayer(DeviceArena& arena, const Config& config,
                                  const LayerWeights& layer, std::uint32_t il,
                                  FfnLayerDevice* out, std::string* error);

// ---- Layer loop ----

/// The MLA / FFN sub-layer operators and their sizing, with the signatures of
/// the layer_ops.hpp functions. Production binds Lane B's (LaneBOps below);
/// tests bind stand-ins. `mla` and `mla_cache_bytes` are required when the
/// uploaded range has an MLA layer, `ffn` always; `op_scratch_bytes` may be
/// empty (no operator scratch beyond the KDA mixer's).
struct LayerOps {
  std::function<std::size_t(const Config&, std::uint32_t)> mla_cache_bytes;
  std::function<std::size_t(const Config&, std::uint32_t)> op_scratch_bytes;
  std::function<bool(const OpContext&, const Config&, std::uint32_t,
                     const MlaLayerDevice&, MlaCache&, const float*,
                     std::uint32_t, std::uint32_t, float*)>
      mla;
  std::function<bool(const OpContext&, const Config&, std::uint32_t,
                     const FfnLayerDevice&, const float*, std::uint32_t,
                     float*)>
      ffn;
};

// ---- PRODUCTION BINDING (Lane B) ----
// MlaForward / FfnForward / MlaCacheBytes / OpScratchBytes are declared in
// layer_ops.hpp but not yet defined (job #647). Once Lane B's sources define
// them, add GLM53_LANE_B_OPS to gufo_glm53_flash's compile definitions and
// pass LaneBOps() to Executor::Create. Until then nothing references them,
// so the library links without Lane B.
#if defined(GLM53_LANE_B_OPS)
[[nodiscard]] LayerOps LaneBOps();
#endif

/// Device weights of one decoder layer. Only the mixer of the layer's kind is
/// filled.
struct LayerDevice {
  struct Hc {
    const float* fn{nullptr};     ///< [24][hc_dim]
    const float* base{nullptr};   ///< [24]
    const float* scale{nullptr};  ///< [3]
  };
  Hc hc_attn, hc_ffn;
  const float* attn_norm{nullptr};  ///< [hidden]
  const float* ffn_norm{nullptr};   ///< [hidden]
  KdaLayerDevice kda;
  MlaLayerDevice mla;
  FfnLayerDevice ffn;
};

/// What Executor::Create uploads and how much it allocates.
struct ExecutorOptions {
  std::uint32_t first_layer{0};
  std::uint32_t last_layer{0};  ///< Exclusive; 0 = every trunk layer.
  bool embed{true};             ///< token_embd; needs first_layer == 0.
  bool head{true};              ///< output_norm and, if bound, output.weight.
  /// Tokens per pass (activation and operator scratch). Longer calls run in
  /// consecutive passes, which is exact (state carries across passes).
  std::uint32_t max_tokens{64};
  std::uint32_t capacity{4096};  ///< Positions per MLA cache.
};

/// Host copies of a Forward's results.
struct ForwardResult {
  std::vector<float> hidden;   ///< [T][hidden] final_norm.out (head only)
  std::vector<float> logits;   ///< [T][vocab] when requested and bound
  std::vector<float> streams;  ///< [T][hc][hidden] after the last layer run
};

/// Layer index reported to taps outside any layer ("hc_head.out",
/// "final_norm.out"; ReferenceModel uses -1).
inline constexpr std::uint32_t kNoLayer = UINT32_MAX;

/// GPU forward over a layer range of one sequence, with the structure of
/// ReferenceModel::Forward: 4 embedding copies, per layer [HcPre -> RMSNorm
/// -> mixer -> HcPost, then the same with the FFN], mean head, final RMSNorm,
/// output. Owns its stream, weights, per-sequence state (KDA state per KDA
/// layer, an MlaCache of MlaCacheBytes per MLA layer) and scratch.
///
/// Taps (when `tap` is set) use ReferenceModel's names, so a probe can diff
/// per layer: in, attn_hc.out0 (post [hc]), attn_hc.out1 (comb [hc*hc],
/// [src][dst]), attn_hc.out2, input_layernorm.out, self_attn.out, ffn_hc.in,
/// ffn_hc.out0/1/2, post_attention_layernorm.out, mlp.out, out0, then
/// hc_head.out and final_norm.out (layer kNoLayer). Stream taps (in,
/// ffn_hc.in, out0) are ne0 = hidden, ne1 = hc, ne2 = tokens; the others
/// ne0 = width, ne1 = tokens. The executor synchronises its stream before
/// each of its own taps; operator taps (e.g. "kda_out") follow layer_ops.hpp.
class Executor {
public:
  /// Uploads [first_layer, last_layer) and the requested embedding / head,
  /// allocates state and scratch. nullptr (with `error`) on a bad range, a
  /// missing operator, an upload or an allocation failure.
  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const ModelWeights& weights, const ExecutorOptions& options, LayerOps ops,
      std::string* error);

  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  /// Zeroes every KDA state and MLA cache; the next call is a fresh prefill.
  /// Also clears a failed state. False if the reset could not be enqueued.
  [[nodiscard]] bool Reset();
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  /// True after a call failed partway: state no longer matches Position(),
  /// so calls are refused until Reset().
  [[nodiscard]] bool Failed() const noexcept { return failed_; }
  [[nodiscard]] std::size_t DeviceBytes() const noexcept {
    return arena_.bytes();
  }
  [[nodiscard]] const LayerDevice& Layer(std::uint32_t il) const {
    return layers_.at(il);
  }
  [[nodiscard]] Stream stream() const noexcept { return stream_; }

  /// Runs `tokens` at positions Position() .. Position()+T-1 and advances.
  /// Needs the embedding. Fills `out->hidden` and (with `want_logits` and
  /// output.weight bound) `out->logits` when the head is uploaded, else
  /// `out->streams`. Synchronous.
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             bool want_logits, ForwardResult* out,
                             const TapFn* tap, std::string* error);

  /// As Forward, from host residual streams [T][hc][hidden] entering
  /// first_layer instead of token ids (a layer range starting past 0).
  [[nodiscard]] bool ForwardStreams(const float* streams,
                                    std::uint32_t n_tokens, bool want_logits,
                                    ForwardResult* out, const TapFn* tap,
                                    std::string* error);

private:
  Executor(const ModelWeights& weights, const ExecutorOptions& options,
           LayerOps ops);
  bool Init(std::string* error);
  bool Run(const std::int32_t* tokens, const float* streams,
           std::uint32_t n_tokens, bool want_logits, ForwardResult* out,
           const TapFn* tap, std::string* error);
  bool Pass(const std::int32_t* tokens, const float* streams, std::uint32_t n,
            bool want_logits, ForwardResult* out, const TapFn* tap,
            std::string* error);
  bool HalfLayer(std::uint32_t il, bool ffn, std::uint32_t n,
                 std::uint32_t pos0, const TapFn* tap);
  void Emit(const TapFn* tap, std::string_view name, std::uint32_t layer,
            const float* data, std::uint64_t ne0, std::uint64_t ne1,
            std::uint64_t ne2 = 1);

  const ModelWeights& weights_;
  const Config& config_;
  ExecutorOptions options_;
  LayerOps ops_;
  DeviceArena arena_;
  Stream stream_{};
  std::vector<LayerDevice> layers_;  ///< Indexed by layer; filled in range.
  std::vector<KdaState> kda_;        ///< Indexed by layer; KDA layers.
  std::vector<MlaCache> mla_;        ///< Indexed by layer; MLA layers.
  DeviceMatrix embd_;
  DeviceMatrix output_;
  const float* output_norm_{nullptr};
  // Activations of one pass, [max_tokens] rows each.
  float* streams_{nullptr};   ///< [T][hc][hidden]
  float* x_{nullptr};         ///< HcPre collapse, then the embedding row
  float* xn_{nullptr};        ///< RMS-normalised sub-layer input
  float* a_{nullptr};         ///< sub-layer output; head mean
  float* split_{nullptr};     ///< [T][24] pre | post | comb
  float* tap_post_{nullptr};  ///< [T][hc], taps only
  float* tap_comb_{nullptr};  ///< [T][hc*hc], taps only
  float* logits_{nullptr};    ///< [T][vocab]
  std::uint32_t* ids_{nullptr};
  void* scratch_{nullptr};
  std::size_t scratch_bytes_{0};
  std::uint32_t position_{0};
  bool failed_{false};
};

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_EXECUTOR_HPP_
