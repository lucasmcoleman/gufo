#ifndef GUFO_MODELS_LAGUNA_REFERENCE_HPP_
#define GUFO_MODELS_LAGUNA_REFERENCE_HPP_

// CPU reference forward ("oracle") for Laguna-S-2.1, written exactly as
// docs/models/laguna-s-2.1/ARCH.md specifies and with llama.cpp's arithmetic
// where ARCH.md names it (ggml RoPE/YaRN, RMSNorm, softplus, the 2^-14 router
// clamp). It is a correctness reference, not a fast path: full recompute of
// the whole sequence, no KV cache.
//
// Weights stay in the (mmapped) GGUF. Each matrix row is dequantized on the
// fly into a small per-thread buffer and consumed immediately, so the 74 GB
// model is never held in F32. MoE layers dequantize only the experts the
// router selected for the tokens at hand.
//
// Two modes, both kept:
//  - F32: activations stay exact F32 (dequantized weights, exact inputs).
//  - Q8-aware: every matmul input is rounded the way the llama.cpp vec-dot
//    path quantizes it for the weight type, so kernel arithmetic bugs are not
//    confused with expected quantization error. See ActivationQuant.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna::reference {

/// How matmul inputs are rounded before the dot product.
enum class ActivationQuant {
  /// Exact F32 inputs.
  kNone,
  /// ggml-cpu `vec_dot_type` of the weight: Q8_0 (32-blocks, fp16 scale) for
  /// Q8_0/IQ4_NL weights, Q8_K (256-blocks, float scale) for Q3_K..Q6_K and
  /// IQ4_XS weights, F16 for F16 weights, BF16 for BF16 weights; F32 weights
  /// get exact inputs. Reference: quantize_row_q8_0_ref /
  /// quantize_row_q8_K_ref.
  kCpu,
  /// ggml HIP/CUDA mul_mat_vec_q path: Q8_1 (32-blocks, fp16 scale, same
  /// rounding as Q8_0) for every quantized weight type; F32 weights exact.
  kGpuQ8_1,
  /// ggml HIP/CUDA mul_mat_q (tile) path: quantize_mmq_q8_1 with 32-element
  /// groups, q = round(x * (127 / amax)), and the stored scale per weight
  /// type's layout: fp16 for Q4_K/Q5_K (DS4 layout), a float d = amax / 127
  /// for Q8_0, Q6_K and IQ4_XS (D4 layout, no fp16 rounding). F32 weights
  /// exact. Q4_K/Q5_K weights also round their per-sub-block scales and mins
  /// to fp16 and use the fp16 block sum, as the tile kernels do; MatMulRows
  /// models that (RoundActivations alone cannot).
  kGpuMmq,
};

/// Router weight denominator. Production gufo follows llama.cpp.
enum class RouterDenominator {
  kLlamaClamp,  ///< w = s[sel] / max(sum, 2^-14)  (llama.cpp d854d98).
  kHfRaw,       ///< w = s[sel] / sum               (modeling_laguna.py:182).
};

/// Operator-test hook (see tests/models/laguna/ffn_gpu.hpp): shifts one fp16
/// quantity of activation row 0 by `steps` fp16 steps, to ask whether a GPU
/// result is the oracle's with one stored scale rounded the other way. The
/// GPU forms block scales with approximate reciprocals, so a scale within an
/// ulp of an fp16 rounding boundary can round the other way. `kind` 0 shifts
/// the stored scale of 32-element group `group` (Q8_1 modes; MMQ Q4_K/Q5_K),
/// kind 1 the stored block sum (MMQ Q4_K/Q5_K only). Inactive when group < 0.
struct ScaleNudge {
  int group{-1};
  int kind{0};
  int steps{0};
};

struct Options {
  ActivationQuant activation_quant{ActivationQuant::kNone};
  ScaleNudge nudge;  ///< Test hook; leave default in production.
  RouterDenominator router_denominator{RouterDenominator::kLlamaClamp};
  /// Round K (after RoPE) and V through F16 before attention, as llama.cpp's
  /// default F16 KV cache does.
  bool round_kv_f16{false};
  /// Compute result_output only for the last token (the other rows are not
  /// returned). Saves most of the output projection for long prompts.
  bool last_row_logits_only{false};
  /// Worker threads for matmul rows. 0 picks the hardware concurrency. The
  /// result does not depend on the thread count.
  unsigned threads{0};
};

enum class TapType { kF32, kI32 };

/// One named intermediate tensor. `name` carries llama.cpp's "-<layer>" suffix
/// for per-layer tensors ("Qcur-3") and none for result_norm/result_output.
/// `ne` follows ggml: ne[0] is the contiguous (fastest) dimension and the
/// token axis is last, so a [width, T] tensor is T rows of `width` values.
struct Tap {
  std::string_view name;
  TapType type{TapType::kF32};
  std::array<std::uint64_t, 3> ne{1, 1, 1};
  const void* data{nullptr};
  [[nodiscard]] std::uint64_t Count() const noexcept {
    return ne[0] * ne[1] * ne[2];
  }
};
using TapFn = std::function<void(const Tap&)>;

/// Writes every tap as `<name>.f32` / `<name>.i32` (raw little-endian) and,
/// on Finish(), `index.json` listing name, file, dtype and ggml `ne`.
class TapWriter {
public:
  /// `filter` keeps only taps whose name contains one of the substrings
  /// (empty keeps all).
  explicit TapWriter(std::filesystem::path directory,
                     std::vector<std::string> filter = {});
  void operator()(const Tap& tap);
  /// Writes index.json. Returns false (message in `error`) on I/O failure.
  bool Finish(std::string* error = nullptr);
  [[nodiscard]] std::size_t written() const noexcept { return entries_.size(); }

private:
  struct Entry {
    std::string name;
    std::string file;
    TapType type;
    std::array<std::uint64_t, 3> ne;
  };
  std::filesystem::path directory_;
  std::vector<std::string> filter_;
  std::vector<Entry> entries_;
  std::string error_;
};

// ---- Building blocks, public so the unit tests can exercise them alone. ----

/// y[t][r] = W[expert][r] . x[t] for `n` rows `x[t]` of `w.cols` values, the
/// oracle's matrix product (inputs rounded per `options.activation_quant`;
/// `y` is [n][w.rows]). Public so operator tests can check single stages.
void MatMulRows(const TensorRef& w, std::uint64_t expert, const float* x,
                std::size_t n, float* y, const Options& options);

/// Round-trip through IEEE binary16 (round to nearest even).
[[nodiscard]] float RoundToF16(float value) noexcept;

/// Rounds `rows` x `cols` activations in place the way `mode` quantizes the
/// matmul input for a weight of type `weight_type`.
void RoundActivations(core::GgmlType weight_type, ActivationQuant mode,
                      float* x, std::size_t rows, std::size_t cols,
                      const ScaleNudge* nudge = nullptr);

/// Dequantizes one logical row of `cols` elements (F32/F16/BF16/Q8_0/Q4_K/
/// Q5_K/Q6_K/IQ4_XS). Aborts on any other type.
void DequantizeRow(core::GgmlType type, const void* src, float* dst,
                   std::size_t cols);

/// ggml_rope_ext parameters of one layer, NeoX layout, leading `n_rot`
/// channels rotated.
struct RopeParams {
  std::uint32_t n_rot{0};
  float freq_base{10000.0F};
  float freq_scale{1.0F};  ///< 1 / factor with YaRN, else 1.
  float ext_factor{0.0F};  ///< 1 with YaRN, 0 without.
  float attn_factor{
      1.0F};  ///< ggml's `attn_factor` argument (not the net mscale).
  float beta_fast{32.0F};
  float beta_slow{1.0F};
  std::uint32_t n_ctx_orig{0};
};
/// Parameters llama.cpp passes for layer `il` (full layers: YaRN; sliding
/// layers: plain RoPE). With YaRN, ggml multiplies cos/sin by
/// attn_factor * (1 + 0.1 ln(1/freq_scale)), which equals Config::YarnMscale().
[[nodiscard]] RopeParams RopeFor(const Config& config, std::uint32_t il);
/// Host-side constants of ggml's rope cache for one layer, in the float
/// arithmetic ApplyRope uses (also consumed by the GPU attention operator so
/// both sides start from the same bits). Per pair i0/2 the angle is the
/// position multiplied i0/2 times by `theta_scale`; with YaRN it is blended
/// between interpolated and extrapolated by a ramp over `corr`, and cos/sin
/// are multiplied by `mscale` (attn_factor times 1 + 0.1 ln(1/freq_scale)
/// with YaRN).
struct RopeSetup {
  float theta_scale{1.0F};
  float corr[2]{0.0F, 0.0F};
  float mscale{1.0F};
};
[[nodiscard]] RopeSetup MakeRopeSetup(const RopeParams& rope);
/// Rotates `heads` heads of `head_dim` channels per token in place. `x` holds
/// `positions.size()` rows of `heads * head_dim`.
void ApplyRope(float* x, std::span<const std::int32_t> positions,
               std::uint32_t heads, std::uint32_t head_dim,
               const RopeParams& rope);

/// Causal softmax attention with grouped-query heads. q is [T][heads*hd],
/// k and v are [T][kv_heads*hd]. Query head h reads kv head h / (heads /
/// kv_heads). `window` 0 means full causal; otherwise key j is visible from
/// position p iff j <= p and p - j < window.
void GqaAttention(const float* q, const float* k, const float* v,
                  std::size_t tokens, std::uint32_t heads,
                  std::uint32_t kv_heads, std::uint32_t head_dim,
                  std::uint32_t window, float scale, float* out);

/// GqaAttention with explicit absolute positions: query row i sits at
/// `q_positions[i]`, key row j at `k_positions[j]`, and key j is visible from
/// query i iff k_pos <= q_pos and (window == 0 or q_pos - k_pos < window).
/// q is [Q][heads*hd], k and v are [K][kv_heads*hd]; the keys need not be
/// sorted. A query with no visible key yields zeros. GqaAttention is this with
/// positions 0..T-1 for both.
void GqaAttentionAt(const float* q, std::span<const std::int32_t> q_positions,
                    const float* k, const float* v,
                    std::span<const std::int32_t> k_positions,
                    std::uint32_t heads, std::uint32_t kv_heads,
                    std::uint32_t head_dim, std::uint32_t window, float scale,
                    float* out);

/// The oracle's RMSNorm row (double sum, float scale) times `weight`.
void RmsNormVector(const float* x, const float* weight, std::size_t n,
                   float eps, float* y);
/// A whole tensor dequantized to F32.
[[nodiscard]] std::vector<float> DequantizeTensor(const TensorRef& t);
/// log(1 + exp(x)), identity above 20, as llama.cpp's softplus.
[[nodiscard]] float SoftplusF32(float x);

/// Routing of one token. `probs` and `weights` are in selection order.
struct Routing {
  std::vector<std::uint32_t> ids;
  std::vector<float> probs;    ///< s[sel], unbiased sigmoid.
  std::vector<float> weights;  ///< Final weights, scale included.
  float sum{0.0F};             ///< sum of probs.
  float denominator{1.0F};     ///< value the sum was divided by.
};
/// Sigmoid routing with a selection-only bias. Ties pick the lower expert id.
/// `normalize` divides by the denominator, `scale` multiplies afterwards
/// (skipped when 0 or 1, as llama.cpp does).
[[nodiscard]] Routing Route(const float* logits, const float* bias,
                            std::uint32_t experts, std::uint32_t used,
                            bool normalize, float scale,
                            RouterDenominator denominator);

/// The MLP of one layer over its post-`ffn_norm` rows `x` ([T][hidden]):
/// dense SwiGLU when `layer.moe` is false, else router, routed experts and
/// shared expert. Returns T rows of `hidden` values. With `tap`, emits the
/// "ffn_*" taps `Model::Forward` emits (suffix "-<layer_index>"; no suffix
/// when `layer_index` is negative). Model::Forward calls this, so the
/// per-layer GPU operator tests compare against exactly the oracle's MLP.
[[nodiscard]] std::vector<float> FfnForward(const Config& config,
                                            const LayerWeights& layer,
                                            const float* x, std::size_t tokens,
                                            const Options& options,
                                            const TapFn& tap = {},
                                            int layer_index = -1);

/// The attention block of one layer over its post-`attn_norm` rows `n`
/// ([T][hidden], positions 0..T-1, full recompute): q/k/v and gate projections,
/// per-head QK RMSNorm, NeoX RoPE (YaRN on full layers), F16 K/V rounding when
/// `options.round_kv_f16`, causal (sliding on SWA layers) GQA attention, the
/// softplus output gate and the output projection. Returns T rows of `hidden`
/// values (before the residual add). With `tap`, emits the attention taps
/// `Model::Forward` emits ("Qcur", ..., "attn_o_proj", suffix "-<layer_index>"
/// or none when negative). Model::Forward calls this.
[[nodiscard]] std::vector<float> AttnForward(
    const Config& config, const LayerWeights& layer, std::uint32_t il,
    const float* n, std::size_t tokens, const Options& options,
    const TapFn& tap = {}, int layer_index = -1);

class Model {
public:
  /// `weights` must outlive the model (it points into the mapping).
  explicit Model(const ModelWeights& weights, Options options = {});

  /// Runs the whole network over `tokens` (positions 0..T-1) and returns the
  /// logits, T rows of vocab_size (or one row with last_row_logits_only).
  /// Emits taps when `tap` is set.
  [[nodiscard]] std::vector<float> Forward(std::span<const std::int32_t> tokens,
                                           const TapFn& tap = {}) const;

private:
  const ModelWeights& weights_;
  Options options_;
};

/// Residual-stream rows of selected target layers, for the DFlash drafter.
struct TargetTaps {
  std::vector<std::uint32_t> layers;  ///< Zero-based layers whose OUTPUT.
  std::size_t hidden_size{0};
  /// [T][layers.size() * hidden]: for each token the taps concatenated in
  /// `layers` order, tap k at offset k * hidden (the drafter's input layout).
  std::vector<float> features;
  std::vector<float> logits;  ///< The same forward's result_output.
};

/// Runs `model` over `tokens` and gathers the `l_out-N` tap (the residual
/// stream after the residual add of zero-based layer N) for each entry of
/// `output_layers`. For the Laguna drafter pass DFlashConfig::tap_layer(i) for
/// every tap, i.e. 1, 10, 19, 29, 38, 47; layer 47 (the last) is the stream
/// before the final norm. Throws when a layer has no tap.
[[nodiscard]] TargetTaps ForwardWithTaps(
    const Model& model, std::span<const std::int32_t> tokens,
    std::span<const std::uint32_t> output_layers);

}  // namespace gufo::models::laguna::reference

#endif  // GUFO_MODELS_LAGUNA_REFERENCE_HPP_
