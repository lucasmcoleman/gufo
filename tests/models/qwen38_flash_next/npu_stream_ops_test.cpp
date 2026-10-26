// GPU checks (no NPU involved) of the kernels behind the two-bank NPU prefill
// and the F16 residual, ported from mikealanni/gufo (MIT, Mike Alani):
// reading NPU C streams directly (NpuDecodeStreamRows, SsmConv4Streams), the
// key/value half of the attention preparation (PrepareKvHeads), the chunked
// gated delta rule against the row-split scan, and the F16 residual combine
// and conversions. Synthetic C streams stand in for NPU output.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

/// Device memory holding a host vector's bytes.
template<typename T>
class Device {
public:
  explicit Device(std::size_t count) : count_(count) {
    CheckHip(hipMalloc(&data_, std::max<std::size_t>(1, count) * sizeof(T)),
             "hipMalloc");
  }
  explicit Device(const std::vector<T>& host) : Device(host.size()) {
    CheckHip(hipMemcpy(data_, host.data(), host.size() * sizeof(T),
                       hipMemcpyHostToDevice),
             "upload");
  }
  ~Device() { (void)hipFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  [[nodiscard]] T* get() const noexcept { return static_cast<T*>(data_); }
  [[nodiscard]] std::vector<T> Read() const {
    std::vector<T> host(count_);
    CheckHip(hipDeviceSynchronize(), "synchronize");
    CheckHip(hipMemcpy(host.data(), data_, count_ * sizeof(T),
                       hipMemcpyDeviceToHost),
             "download");
    return host;
  }

private:
  void* data_{nullptr};
  std::size_t count_{0};
};

constexpr std::uint32_t kStreamCols = 1024;
constexpr std::uint32_t kStreamTokens = 512;  // one 512-row C tile row

/// Random C streams: groups x slices streams of kStreamTokens x 1024 values,
/// each 9-byte group an exponent near 127 and eight int8 mantissas.
std::vector<std::uint8_t> RandomStreams(std::uint32_t streams,
                                        std::uint32_t seed) {
  const std::size_t bytes = std::size_t{kStreamTokens} * kStreamCols / 8 * 9;
  std::vector<std::uint8_t> data(bytes * streams);
  for (std::size_t g = 0; g < data.size() / 9; ++g) {
    data[g * 9] = static_cast<std::uint8_t>(120 + NextRandom(&seed) % 12);
    for (int i = 1; i < 9; ++i) {
      data[g * 9 + i] = static_cast<std::uint8_t>(NextRandom(&seed));
    }
  }
  return data;
}

/// The device table of stream pointers, [group * stride + slice].
std::vector<const std::uint8_t*> StreamTable(const std::uint8_t* base,
                                             std::uint32_t streams) {
  const std::size_t bytes = std::size_t{kStreamTokens} * kStreamCols / 8 * 9;
  std::vector<const std::uint8_t*> table(streams);
  for (std::uint32_t i = 0; i < streams; ++i) {
    table[i] = base + i * bytes;
  }
  return table;
}

/// Decodes the streams with the PR #568 decoder into [tokens][stride * 1024].
std::vector<float> DecodeAll(const std::vector<const std::uint8_t*>& table,
                             std::uint32_t groups, std::uint32_t stride) {
  const std::uint32_t cols = stride * kStreamCols;
  Device<float> out(std::size_t{groups} * kStreamTokens * cols);
  for (std::uint32_t g = 0; g < groups; ++g) {
    for (std::uint32_t s = 0; s < stride; ++s) {
      q::NpuDecodeOutput(table[g * stride + s],
                         out.get() + std::size_t{g} * kStreamTokens * cols,
                         kStreamTokens, kStreamCols, cols, s * kStreamCols,
                         false, nullptr);
    }
  }
  return out.Read();
}

void CheckStreamRows() {
  constexpr std::uint32_t kGroups = 2;
  constexpr std::uint32_t kStride = 3;
  const auto host = RandomStreams(kGroups * kStride, 0x5157E4A1U);
  Device<std::uint8_t> streams(host);
  const auto table = StreamTable(streams.get(), kGroups * kStride);
  Device<const std::uint8_t*> d_table(table);
  const auto all = DecodeAll(table, kGroups, kStride);
  const std::uint32_t cols = kStride * kStreamCols;
  // Rows straddling the group boundary, into a narrower pitch from column 5.
  constexpr std::uint32_t kRow0 = kStreamTokens - 3;
  constexpr std::uint32_t kRows = 7;
  constexpr std::uint32_t kCol0 = 5;
  constexpr std::uint32_t kPitch = cols + 16;
  Device<float> out(std::size_t{kRows} * kPitch);
  CheckHip(hipMemset(out.get(), 0, std::size_t{kRows} * kPitch * 4), "zero");
  q::NpuDecodeStreamRows(d_table.get(), kStride, kStreamTokens, out.get(),
                         kRow0, kRows, cols - kCol0, kPitch, kCol0, nullptr);
  const auto rows = out.Read();
  for (std::uint32_t r = 0; r < kRows; ++r) {
    for (std::uint32_t c = 0; c + kCol0 < cols; ++c) {
      Require(rows[std::size_t{r} * kPitch + kCol0 + c] ==
                  all[std::size_t{kRow0 + r} * cols + c],
              "NpuDecodeStreamRows differs from NpuDecodeOutput");
    }
  }
  std::cout << "stream rows: match\n";
}

float Silu(float x) {
  return x / (1.0F + std::exp(-x));
}

void CheckStreamConv(std::uint32_t n_tokens) {
  constexpr std::uint32_t kStride = 3;  // 3 slices = 3072 channels
  constexpr std::uint32_t kChannels = kStride * kStreamCols;
  const std::uint32_t groups = (n_tokens + kStreamTokens - 1) / kStreamTokens;
  const auto host = RandomStreams(groups * kStride, 0xC0417E5U + n_tokens);
  Device<std::uint8_t> streams(host);
  const auto table = StreamTable(streams.get(), groups * kStride);
  Device<const std::uint8_t*> d_table(table);
  const auto x = DecodeAll(table, groups, kStride);
  std::uint32_t seed = 0x7A95U;
  std::vector<float> w(std::size_t{kChannels} * 4);
  std::vector<float> history(std::size_t{3} * kChannels);
  for (float& v : w) {
    v = Uniform(&seed, 0.5F);
  }
  for (float& v : history) {
    v = Uniform(&seed, 1.0F);
  }
  Device<float> d_w(w);
  Device<float> d_history(history);
  Device<float> out(std::size_t{n_tokens} * kChannels);
  q::SsmConv4Streams(d_table.get(), kStride, kStreamTokens, d_w.get(),
                     d_history.get(), out.get(), n_tokens, kChannels, nullptr,
                     false);
  const auto f32 = out.Read();
  double worst = 0.0;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    for (std::uint32_t c = 0; c < kChannels; ++c) {
      double acc = 0.0;
      for (std::uint32_t k = 0; k < 4; ++k) {
        const int src = static_cast<int>(t + k) - 3;
        const double v =
            src >= 0
                ? x[static_cast<std::size_t>(src) * kChannels + c]
                : history[static_cast<std::size_t>(3 + src) * kChannels + c];
        acc += w[std::size_t{c} * 4 + k] * v;
      }
      const double ref = Silu(static_cast<float>(acc));
      const double got = f32[std::size_t{t} * kChannels + c];
      worst =
          std::max(worst, std::abs(got - ref) / std::max(1e-2, std::abs(ref)));
    }
  }
  Require(worst < 1e-4, "SsmConv4Streams differs from the CPU convolution: " +
                            std::to_string(worst));
  // The history buffer is read, not advanced.
  Require(d_history.Read() == history, "SsmConv4Streams wrote its history");
  if (n_tokens % 32 == 0) {
    // The F16 output is the F32 result rounded (the same kernel computes
    // both).
    Device<__half> half(std::size_t{n_tokens} * kChannels);
    q::SsmConv4Streams(d_table.get(), kStride, kStreamTokens, d_w.get(),
                       d_history.get(), half.get(), n_tokens, kChannels,
                       nullptr, true);
    const auto h = half.Read();
    for (std::size_t i = 0; i < h.size(); ++i) {
      const __half expected = __float2half_rn(f32[i]);
      Require(std::memcmp(&expected, &h[i], sizeof(__half)) == 0,
              "SsmConv4Streams F16 output is not the rounded F32 output");
    }
  }
  std::cout << "stream conv n=" << n_tokens << ": worst " << worst << "\n";
}

void CheckKvHeads() {
  constexpr std::uint32_t kHeads = 24;
  constexpr std::uint32_t kKvHeads = 2;
  constexpr std::uint32_t kD = 256;
  constexpr std::uint32_t kStride = 2 * kHeads * kD + 2 * kKvHeads * kD;
  constexpr std::uint32_t kTokens = 96;
  constexpr std::uint32_t kStart = 37;
  std::uint32_t seed = 0x4B5A11U;
  std::vector<float> packed(std::size_t{kTokens} * kStride);
  for (float& v : packed) {
    v = Uniform(&seed, 2.0F);
  }
  std::vector<float> k_gamma(kD);
  std::vector<float> q_gamma(kD);
  for (float& v : k_gamma) {
    v = 1.0F + Uniform(&seed, 0.25F);
  }
  for (float& v : q_gamma) {
    v = 1.0F + Uniform(&seed, 0.25F);
  }
  Device<float> d_packed(packed);
  Device<float> d_kg(k_gamma);
  Device<float> d_qg(q_gamma);
  Device<std::uint32_t> d_pos(std::vector<std::uint32_t>{kStart});
  const std::size_t cache = std::size_t{kStart + kTokens} * kKvHeads * kD;
  const std::size_t qcount = std::size_t{kTokens} * kHeads * kD;
  Device<float> q(qcount);
  Device<float> gate(qcount);
  Device<__half> k_ref(cache);
  Device<__half> v_ref(cache);
  Device<__half> k_got(cache);
  Device<__half> v_got(cache);
  for (const auto* d : {&k_ref, &v_ref, &k_got, &v_got}) {
    CheckHip(hipMemset(d->get(), 0, cache * sizeof(__half)), "zero cache");
  }
  Require(q::PrepareAttention(d_packed.get(), kStride, d_qg.get(), d_kg.get(),
                              q.get(), gate.get(), k_ref.get(), v_ref.get(),
                              kTokens, kHeads, kKvHeads, kD, 64, d_pos.get(),
                              1e7F, 1e-6F, nullptr, nullptr, true),
          "PrepareAttention rejected the geometry");
  Require(q::PrepareKvHeads(d_packed.get(), kStride, d_kg.get(), k_got.get(),
                            v_got.get(), kTokens, kHeads, kKvHeads, kD, 64,
                            d_pos.get(), 1e7F, 1e-6F, nullptr, nullptr),
          "PrepareKvHeads rejected the geometry");
  const auto kr = k_ref.Read();
  const auto vr = v_ref.Read();
  const auto kg = k_got.Read();
  const auto vg = v_got.Read();
  Require(std::memcmp(kr.data(), kg.data(), cache * sizeof(__half)) == 0,
          "PrepareKvHeads keys differ from PrepareAttention");
  Require(std::memcmp(vr.data(), vg.data(), cache * sizeof(__half)) == 0,
          "PrepareKvHeads values differ from PrepareAttention");
  std::cout << "kv heads: match\n";
}

/// Chunked (F16-operand WY) recurrence against the row-split scan on the
/// same F16-rounded convolution rows.
void CheckChunkedGdn() {
  constexpr std::uint32_t kKHeads = 16;
  constexpr std::uint32_t kVHeads = 48;
  constexpr std::uint32_t kDim = 128;
  constexpr std::uint32_t kChannels = 2 * kKHeads * kDim + kVHeads * kDim;
  constexpr std::uint32_t kZ = kVHeads * kDim;
  constexpr std::uint32_t kTokens = 256;
  Require(q::GdnChunkedEligible(kTokens, kKHeads, kVHeads, kDim),
          "chunked recurrence rejects the Flash-Next geometry");
  Require(!q::GdnChunkedEligible(kTokens + 1, kKHeads, kVHeads, kDim),
          "negative: a partial chunk was accepted");
  std::uint32_t seed = 0x6D17C4U;
  std::vector<float> conv(std::size_t{kTokens} * kChannels);
  for (float& v : conv) {
    v = __half2float(__float2half_rn(Uniform(&seed, 1.0F)));
  }
  std::vector<float> z(std::size_t{kTokens} * kZ);
  for (float& v : z) {
    v = Uniform(&seed, 2.0F);
  }
  std::vector<float> ab(std::size_t{kTokens} * 2 * kVHeads);
  for (float& v : ab) {
    v = Uniform(&seed, 2.0F);
  }
  std::vector<float> a(kVHeads);
  std::vector<float> dt(kVHeads);
  std::vector<float> norm(kDim);
  for (float& v : a) {
    v = -1.5F + Uniform(&seed, 1.0F);
  }
  for (float& v : dt) {
    v = Uniform(&seed, 1.0F);
  }
  for (float& v : norm) {
    v = 1.0F + Uniform(&seed, 0.5F);
  }
  std::vector<float> conv_w(std::size_t{kChannels} * 4, 0.0F);
  std::vector<float> history(std::size_t{3} * kChannels, 0.0F);
  Device<float> d_qkv(conv);  // only the history update reads it
  Device<float> d_z(z);
  Device<float> d_ab(ab);
  Device<float> d_a(a);
  Device<float> d_dt(dt);
  Device<float> d_norm(norm);
  Device<float> d_conv_w(conv_w);
  const std::size_t scratch = std::size_t{kTokens + 4} * kChannels;
  const std::size_t state_count = std::size_t{kVHeads} * kDim * kDim;
  std::vector<float> outs[2];
  std::vector<float> states[2];
  for (int route = 0; route < 2; ++route) {
    const bool chunked = route == 1;
    Device<float> d_scratch(scratch);
    if (chunked) {
      std::vector<__half> half(conv.size());
      for (std::size_t i = 0; i < conv.size(); ++i) {
        half[i] = __float2half_rn(conv[i]);
      }
      CheckHip(hipMemcpy(d_scratch.get(), half.data(),
                         half.size() * sizeof(__half), hipMemcpyHostToDevice),
               "half conv rows");
    } else {
      CheckHip(hipMemcpy(d_scratch.get(), conv.data(),
                         conv.size() * sizeof(float), hipMemcpyHostToDevice),
               "conv rows");
    }
    Device<float> d_history(history);
    Device<float> d_state(std::vector<float>(state_count, 0.0F));
    Device<float> d_qn(std::size_t{kTokens} * kKHeads * kDim);
    Device<float> d_kn(std::size_t{kTokens} * kKHeads * kDim);
    Device<float> d_raw(std::size_t{kTokens} * kVHeads * kDim);
    Device<float> d_out(std::size_t{kTokens} * kZ);
    const std::size_t chunk_bytes = q::GdnChunkScratchBytesFor(kTokens);
    Device<std::uint8_t> d_chunk(chunk_bytes);
    const bool ok = q::GatedDeltaNet(
        d_qkv.get(), kChannels, d_z.get(), kZ, d_ab.get(), d_conv_w.get(),
        d_a.get(), d_dt.get(), d_norm.get(), d_history.get(), d_scratch.get(),
        d_qn.get(), d_kn.get(), d_raw.get(), d_state.get(), d_out.get(),
        nullptr, {}, {}, kTokens, kKHeads, kVHeads, kDim, 4, true, true, 1e-6F,
        nullptr, nullptr, {}, 0, nullptr, nullptr, nullptr, nullptr, 0, 0, 0,
        chunked, chunked ? d_chunk.get() : nullptr, chunked ? chunk_bytes : 0);
    Require(ok, "GatedDeltaNet rejected the route");
    outs[route] = d_out.Read();
    states[route] = d_state.Read();
  }
  const auto rms_rel = [](const std::vector<float>& ref,
                          const std::vector<float>& got) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      num += (static_cast<double>(got[i]) - ref[i]) *
             (static_cast<double>(got[i]) - ref[i]);
      den += static_cast<double>(ref[i]) * ref[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
  };
  const double out_err = rms_rel(outs[0], outs[1]);
  const double state_err = rms_rel(states[0], states[1]);
  std::cout << "chunked GDN: output rel RMS " << out_err << ", state "
            << state_err << "\n";
  // The fork measured 3-4e-4 relative per layer; allow headroom for random
  // inputs while still failing on a wrong chunk boundary or decay.
  Require(out_err < 5e-3 && state_err < 5e-3,
          "chunked recurrence diverges from the row-split scan");
  // Negative: a partial chunk request must be refused without launching.
  Device<float> d_scratch(scratch);
  Device<float> d_state(state_count);
  Device<float> d_out(std::size_t{kTokens} * kZ);
  Require(!q::GatedDeltaNet(d_qkv.get(), kChannels, d_z.get(), kZ, d_ab.get(),
                            d_conv_w.get(), d_a.get(), d_dt.get(), d_norm.get(),
                            d_scratch.get(), d_scratch.get(), d_scratch.get(),
                            d_scratch.get(), d_scratch.get(), d_state.get(),
                            d_out.get(), nullptr, {}, {}, kTokens - 32, kKHeads,
                            kVHeads, kDim, 4, true, true, 1e-6F, nullptr,
                            nullptr, {}, 0, nullptr, nullptr, nullptr, nullptr,
                            0, 0, 0, true, d_scratch.get(), 1),
          "negative: an undersized chunked request was launched");
}

/// The F16 residual combine against the F32 combine fed the same
/// (F16-representable) residual: the norm outputs match bit for bit and the
/// stored residual is the F32 result rounded; stores saturate.
void CheckF16Residual() {
  constexpr std::uint32_t kHidden = 2560;
  constexpr std::uint32_t kStreams = 4;
  constexpr std::uint32_t kTokens = 64;
  constexpr std::size_t kHc = std::size_t{kStreams} * kHidden;
  std::uint32_t seed = 0xF16F16U;
  std::vector<float> res(kTokens * kHc);
  for (float& v : res) {
    v = __half2float(__float2half_rn(Uniform(&seed, 8.0F)));
  }
  // A value near the F16 limit that the update pushes past it (saturates).
  res[17] = __half2float(__float2half_rn(65000.0F));
  std::vector<float> block(kTokens * kHidden);
  for (float& v : block) {
    v = Uniform(&seed, 4.0F);
  }
  block[17] = 30000.0F;
  std::vector<float> inject(kTokens * kStreams);
  for (float& v : inject) {
    v = Uniform(&seed, 2.0F);
  }
  std::vector<float> gamma(kHc);
  for (float& v : gamma) {
    v = 1.0F + Uniform(&seed, 0.25F);
  }
  std::vector<__half> res_h(res.size());
  for (std::size_t i = 0; i < res.size(); ++i) {
    res_h[i] = __float2half_rn(res[i]);
  }
  Device<float> d_res(res);
  Device<__half> d_res_h(res_h);
  Device<float> d_block(block);
  Device<float> d_inject(inject);
  Device<float> d_gamma(gamma);
  Device<__half> xn_f(kTokens * kHc);
  Device<__half> xn_h(kTokens * kHc);
  q::HcCombineF16(d_res.get(), d_block.get(), d_inject.get(), 1, d_gamma.get(),
                  xn_f.get(), nullptr, kTokens, kHidden, kStreams, 1e-6F,
                  nullptr);
  q::HcCombineF16(nullptr, d_block.get(), d_inject.get(), 1, d_gamma.get(),
                  xn_h.get(), nullptr, kTokens, kHidden, kStreams, 1e-6F,
                  nullptr, nullptr, nullptr, nullptr, d_res_h.get());
  const auto rf = d_res.Read();
  const auto rh = d_res_h.Read();
  const auto xf = xn_f.Read();
  const auto xh = xn_h.Read();
  Require(std::memcmp(xf.data(), xh.data(), xf.size() * sizeof(__half)) == 0,
          "F16-residual combine changed the norm");
  for (std::size_t i = 0; i < rf.size(); ++i) {
    const float clamped = std::min(std::max(rf[i], -65504.0F), 65504.0F);
    const __half expected = __float2half_rn(clamped);
    Require(
        std::memcmp(&expected, &rh[i], sizeof(__half)) == 0,
        "F16 residual is not the rounded F32 residual at " + std::to_string(i));
  }
  Require(__half2float(rh[17]) == 65504.0F, "F16 residual did not saturate");
  // Round trip through the conversions.
  Device<__half> half(rf.size());
  Device<float> back(rf.size());
  q::ResidualToHalf(d_res.get(), half.get(), rf.size(), nullptr);
  q::ResidualToFloat(half.get(), back.get(), rf.size(), nullptr);
  const auto hb = half.Read();
  const auto fb = back.Read();
  for (std::size_t i = 0; i < rf.size(); ++i) {
    Require(std::memcmp(&hb[i], &rh[i], sizeof(__half)) == 0,
            "ResidualToHalf differs from the combine's F16 store");
    Require(fb[i] == __half2float(rh[i]), "ResidualToFloat changed a value");
  }
  std::cout << "F16 residual: match\n";
}

}  // namespace

int main() {
  try {
    CheckStreamRows();
    CheckStreamConv(1024);
    CheckStreamConv(1000);  // the per-token fallback kernel
    CheckKvHeads();
    CheckChunkedGdn();
    CheckF16Residual();
  } catch (const std::exception& e) {
    std::cerr << "npu_stream_ops_test: " << e.what() << "\n";
    return 1;
  }
  std::cout << "npu_stream_ops_test: PASS\n";
  return 0;
}
