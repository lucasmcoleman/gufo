#include "src/core/quant/ggml_dequant.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace {

constexpr std::size_t kBlockElements = 256;

void ExpectNear(float actual, float expected, std::string_view message,
                float tolerance = 1.0e-5F) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "Assertion failed: " << message << " actual=" << actual
              << " expected=" << expected << '\n';
    std::exit(1);
  }
}

std::array<std::uint8_t, 144> MakeQ4Block(
    const std::array<std::uint8_t, 8>& scales,
    const std::array<std::uint8_t, 8>& minima) {
  std::array<std::uint8_t, 144> block{};
  block[0] = 0x00;
  block[1] = 0x3C;
  block[2] = 0x00;
  block[3] = 0x38;
  for (std::size_t index = 0; index < 4; ++index) {
    block[4 + index] = static_cast<std::uint8_t>(
        (scales[index] & 0x3FU) | ((scales[index + 4] >> 4U) << 6U));
    block[8 + index] = static_cast<std::uint8_t>(
        (minima[index] & 0x3FU) | ((minima[index + 4] >> 4U) << 6U));
    block[12 + index] = static_cast<std::uint8_t>(
        (scales[index + 4] & 0x0FU) | ((minima[index + 4] & 0x0FU) << 4U));
  }
  for (std::size_t pair = 0; pair < 4; ++pair) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const auto low = static_cast<std::uint8_t>((lane + pair) & 0x0FU);
      const auto high = static_cast<std::uint8_t>((15U - lane + pair) & 0x0FU);
      block[16 + (pair * 32) + lane] =
          static_cast<std::uint8_t>(low | (high << 4U));
    }
  }
  return block;
}

std::array<std::uint8_t, 210> MakeQ6Block(
    const std::array<std::int8_t, 16>& scales) {
  std::array<std::uint8_t, 210> block{};
  const auto logical_quant = [](std::size_t index) {
    return static_cast<std::int8_t>(static_cast<int>((index * 7U) % 64U) - 32);
  };
  for (std::size_t half = 0; half < 2; ++half) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      std::array<std::uint8_t, 4> code{};
      for (std::size_t segment = 0; segment < code.size(); ++segment) {
        code[segment] = static_cast<std::uint8_t>(
            logical_quant((half * 128) + (segment * 32) + lane) + 32);
      }
      block[(half * 64) + lane] = static_cast<std::uint8_t>(
          (code[0] & 0x0FU) | ((code[2] & 0x0FU) << 4U));
      block[(half * 64) + 32 + lane] = static_cast<std::uint8_t>(
          (code[1] & 0x0FU) | ((code[3] & 0x0FU) << 4U));
      block[128 + (half * 32) + lane] = static_cast<std::uint8_t>(
          ((code[0] >> 4U) & 0x03U) | (((code[1] >> 4U) & 0x03U) << 2U) |
          (((code[2] >> 4U) & 0x03U) << 4U) |
          (((code[3] >> 4U) & 0x03U) << 6U));
    }
  }
  std::ranges::copy(scales, block.begin() + 192);
  block[208] = 0x00;
  block[209] = 0x3C;
  return block;
}

std::array<std::uint8_t, 110> MakeQ3Block(
    const std::array<std::int8_t, 16>& scales) {
  std::array<std::uint8_t, 110> block{};
  for (std::size_t index = 0; index < scales.size(); ++index) {
    const auto encoded = static_cast<std::uint8_t>(scales[index] + 32);
    if (index < 8) {
      block[96 + index] |= encoded & 0x0FU;
    } else {
      block[96 + index - 8] |=
          static_cast<std::uint8_t>((encoded & 0x0FU) << 4U);
    }
    block[104 + (index % 4)] |= static_cast<std::uint8_t>(
        ((encoded >> 4U) & 0x03U) << (2U * (index / 4)));
  }

  for (std::size_t index = 0; index < kBlockElements; ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t pair = within_half / 32;
    const std::size_t lane = within_half % 32;
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 3U) % 8U) - 4);
    const auto low = static_cast<std::uint8_t>(quant < 0 ? quant + 4 : quant);
    block[32 + (half * 32) + lane] |=
        static_cast<std::uint8_t>(low << (2U * pair));
    if (quant >= 0) {
      block[lane] |= static_cast<std::uint8_t>(1U << ((half * 4) + pair));
    }
  }
  block[108] = 0x00;
  block[109] = 0x3C;
  return block;
}

void TestQ4() {
  const std::array<std::uint8_t, 8> scales{1, 7, 15, 31, 32, 47, 55, 63};
  const std::array<std::uint8_t, 8> minima{0, 1, 2, 3, 4, 5, 6, 7};
  const auto block = MakeQ4Block(scales, minima);
  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ4_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t group = index / 32;
    const std::size_t pair = group / 2;
    const std::size_t lane = index % 32;
    const auto quant =
        group % 2 == 0 ? static_cast<std::uint8_t>((lane + pair) & 0x0FU)
                       : static_cast<std::uint8_t>((15U - lane + pair) & 0x0FU);
    expected[index] = static_cast<float>(scales[group] * quant) -
                      (0.5F * static_cast<float>(minima[group]));
    ExpectNear(actual[index], expected[index], "Q4_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.5F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ4_K(block.data(), vector, vector.size()),
             expected_dot, "Q4_K dot product");
}

void TestQ6() {
  const std::array<std::int8_t, 16> scales{-8, -7, -6, -5, -4, -3, -2, -1,
                                           1,  2,  3,  4,  5,  6,  7,  8};
  const auto block = MakeQ6Block(scales);
  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ6_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t segment = within_half / 32;
    const std::size_t lane = within_half % 32;
    const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 7U) % 64U) - 32);
    expected[index] =
        static_cast<float>(scales[scale_index]) * static_cast<float>(quant);
    ExpectNear(actual[index], expected[index], "Q6_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.25F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ6_K(block.data(), vector, vector.size()),
             expected_dot, "Q6_K dot product");
}

void TestQ3() {
  const std::array<std::int8_t, 16> scales{-32, -27, -18, -9, -1, 0,  1,  7,
                                           12,  18,  23,  27, 29, 30, 31, -16};
  const auto block = MakeQ3Block(scales);
  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ3_K(block.data(), actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::size_t half = index / 128;
    const std::size_t within_half = index % 128;
    const std::size_t scale_index =
        (half * 8) + ((within_half / 32) * 2) + ((within_half % 32) / 16);
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((index * 3U) % 8U) - 4);
    expected[index] =
        static_cast<float>(scales[scale_index]) * static_cast<float>(quant);
    ExpectNear(actual[index], expected[index], "Q3_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), 0.125F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ3_K(block.data(), vector, vector.size()),
             expected_dot, "Q3_K dot product");
}

void TestQ8() {
  struct Q8Block {
    float d;
    std::int8_t qs[256];
    std::int16_t bsums[16];
  };
  Q8Block block{};
  block.d = 0.00390625F;
  for (std::size_t i = 0; i < kBlockElements; ++i) {
    block.qs[i] = static_cast<std::int8_t>(static_cast<int>(i % 65) - 32);
    block.bsums[i / 16] += block.qs[i];
  }

  std::array<float, kBlockElements> actual{};
  gufo::quant::DequantizeQ8_K(&block, actual.data(), actual.size());

  std::array<float, kBlockElements> expected{};
  for (std::size_t index = 0; index < kBlockElements; ++index) {
    expected[index] = block.d * static_cast<float>(block.qs[index]);
    ExpectNear(actual[index], expected[index], "Q8_K dequant value");
  }

  std::array<float, kBlockElements> vector{};
  std::iota(vector.begin(), vector.end(), -0.75F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ8_K(&block, vector, vector.size()),
             expected_dot, "Q8_K dot product");
}

void TestQ8_0() {
  constexpr std::size_t kQ8_0Elements = 32;
  struct Q8_0Block {
    std::uint16_t d;
    std::int8_t qs[kQ8_0Elements];
  };
  static_assert(sizeof(Q8_0Block) == 34);
  Q8_0Block block{};
  block.d = 0x3800;  // fp16 0.5
  for (std::size_t i = 0; i < kQ8_0Elements; ++i) {
    block.qs[i] = static_cast<std::int8_t>(static_cast<int>(i % 65) - 32);
  }

  std::array<float, kQ8_0Elements> actual{};
  gufo::quant::DequantizeQ8_0(&block, actual.data(), actual.size());

  std::array<float, kQ8_0Elements> expected{};
  for (std::size_t index = 0; index < kQ8_0Elements; ++index) {
    expected[index] =
        gufo::quant::Fp16ToFloat(block.d) * static_cast<float>(block.qs[index]);
    ExpectNear(actual[index], expected[index], "Q8_0 dequant value");
  }

  std::array<float, kQ8_0Elements> vector{};
  std::iota(vector.begin(), vector.end(), -0.75F);
  const float expected_dot = std::inner_product(
      expected.begin(), expected.end(), vector.begin(), 0.0F);
  ExpectNear(gufo::quant::DotProductQ8_0(&block, vector, vector.size()),
             expected_dot, "Q8_0 dot product");
}

void TestQ5_K() {
  constexpr std::size_t kQ5Elements = 256;
  struct Q5Block {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qh[32];
    std::uint8_t qs[128];
  };
  static_assert(sizeof(Q5Block) == 176);

  // QuantizedRowBytes covers every packed projection/embedding format used by
  // Qwen, including both Q8 layouts.
  if (gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ5_K,
                                     kQ5Elements) != 176 ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ8_K, 256) != 292 ||
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ8_0, 32) != 34 ||
      gufo::quant::EncodedSizeBytes(gufo::core::GgmlType::kF32, 8) != 32 ||
      gufo::quant::EncodedSizeBytes(gufo::core::GgmlType::kBF16, 8) != 16 ||
      gufo::quant::EncodedSizeBytes(gufo::core::GgmlType::kQ8_K, 256) != 292) {
    std::cerr << "Assertion failed: QuantizedRowBytes contract\n";
    std::exit(1);
  }

  // Build a handcrafted block. scales[12] is chosen so GetQ4ScaleMin(0..7)
  // yields (scale=4, min=1) for every index: packed[0..3]=4, packed[4..7]=1,
  // packed[8..11]=0x14 (high-half scale/min encodings) -> sc=4, m=1 all 8.
  const auto make_block = [](bool high_bits) {
    Q5Block block{};
    block.d = 0x3400;     // fp16 0.25
    block.dmin = 0x3000;  // fp16 0.125
    for (std::size_t index = 0; index < 4; ++index) {
      block.scales[index] = 4;
      block.scales[index + 4] = 1;
      block.scales[index + 8] = 0x14;
    }
    // qs[gg*32+lane] nibbles: group 0 low nibble=1, group 1 -> 2, etc.
    for (std::size_t gg = 0; gg < 4; ++gg) {
      const auto nibble = static_cast<std::uint8_t>(gg + 1);
      const auto val = static_cast<std::uint8_t>(nibble | (nibble << 4U));
      for (std::size_t lane = 0; lane < 32; ++lane) {
        block.qs[(gg * 32) + lane] = val;
      }
    }
    if (high_bits) {
      for (std::size_t lane = 0; lane < 32; ++lane) {
        block.qh[lane] = 0xFF;
      }
    }
    return block;
  };

  const auto check_block = [](const Q5Block& block, bool high_bits) {
    std::array<float, kQ5Elements> actual{};
    gufo::quant::DequantizeQ5_K(&block, actual.data(), actual.size());

    std::array<float, kQ5Elements> expected{};
    for (std::size_t index = 0; index < kQ5Elements; ++index) {
      const std::size_t gg = index / 64;
      const auto quant = static_cast<float>(gg + 1 + (high_bits ? 16 : 0));
      // Fp16ToFloat(0x3400)=0.25, Fp16ToFloat(0x3000)=0.125, sc=4, m=1.
      expected[index] = quant - 0.125F;
      ExpectNear(actual[index], expected[index],
                 high_bits ? "Q5_K dequant (high bits)" : "Q5_K dequant value");
    }

    std::array<float, kQ5Elements> unit{};
    unit.fill(1.0F);
    const float expected_dot =
        std::accumulate(expected.begin(), expected.end(), 0.0F);
    ExpectNear(gufo::quant::DotProductQ5_K(&block, unit, unit.size()),
               expected_dot, "Q5_K dot product");
  };

  check_block(make_block(false), false);  // no high bits
  check_block(make_block(true), true);    // all high bits set
}

struct ModelSample {
  std::size_t index;
  float value;
};

void TestModelRow(const gufo::core::GgufReader& reader, std::string_view name,
                  gufo::core::GgmlType expected_type, std::size_t expected_k,
                  std::span<const ModelSample> samples, double expected_dot) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr ||
      tensor->dimensions.empty()) {
    std::cerr << "Assertion failed: missing model tensor " << name << '\n';
    std::exit(1);
  }
  if (tensor->type != expected_type ||
      tensor->dimensions.front() != expected_k) {
    std::cerr << "Assertion failed: model tensor contract " << name << '\n';
    std::exit(1);
  }

  std::vector<float> row(expected_k);
  switch (tensor->type) {
    case gufo::core::GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(tensor->data, row.data(), row.size());
      break;
    case gufo::core::GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(tensor->data, row.data(), row.size());
      break;
    default:
      std::cerr << "Assertion failed: unsupported model tensor type " << name
                << '\n';
      std::exit(1);
  }

  for (const auto& sample : samples) {
    ExpectNear(row[sample.index], sample.value, "real GGUF dequant value",
               1.0e-8F);
  }

  std::vector<float> input(expected_k);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
        static_cast<float>(static_cast<int>(index % 257) - 128) / 64.0F;
  }
  const float actual_dot =
      tensor->type == gufo::core::GgmlType::kQ4_K
          ? gufo::quant::DotProductQ4_K(tensor->data, input, input.size())
          : gufo::quant::DotProductQ6_K(tensor->data, input, input.size());
  ExpectNear(actual_dot, static_cast<float>(expected_dot),
             "real GGUF dot product", 5.0e-4F);
}

void TestModelRows(const std::filesystem::path& model_path) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (reader == nullptr) {
    std::cerr << "Assertion failed: MTP GGUF opens: " << error << '\n';
    std::exit(1);
  }

  constexpr std::array<ModelSample, 16> kQ4Samples{{
      {0, 0.008269786834716797F},
      {1, -0.019530773162841797F},
      {15, 0.023714542388916016F},
      {31, -0.01335287094116211F},
      {32, 0.009994983673095703F},
      {63, 0.005248546600341797F},
      {64, -0.002658843994140625F},
      {127, 0.013089179992675781F},
      {128, -0.006126880645751953F},
      {255, -0.0026617050170898438F},
      {256, -0.002628028392791748F},
      {511, 0.013987362384796143F},
      {1023, -0.0033451318740844727F},
      {2047, 0.0010213851928710938F},
      {4095, -0.00015413761138916016F},
      {5119, -0.0033974647521972656F},
  }};
  TestModelRow(*reader, "blk.64.nextn.eh_proj.weight",
               gufo::core::GgmlType::kQ4_K, 10240, kQ4Samples,
               1.1511284159496427);

  constexpr std::array<ModelSample, 16> kQ6Samples{{
      {0, -0.0027894973754882812F},
      {1, -0.013947486877441406F},
      {15, -0.024175643920898438F},
      {31, 0.009633064270019531F},
      {32, -0.003979682922363281F},
      {63, 0.008864402770996094F},
      {64, 0.0030994415283203125F},
      {127, -0.006855964660644531F},
      {128, -0.008070945739746094F},
      {255, 0.011157989501953125F},
      {256, -0.03636360168457031F},
      {511, 0.024633407592773438F},
      {1023, -0.0008414983749389648F},
      {2047, -0.012941122055053711F},
      {4095, 0.02767181396484375F},
      {5119, -0.012220144271850586F},
  }};
  TestModelRow(*reader, "blk.64.attn_q.weight", gufo::core::GgmlType::kQ6_K,
               5120, kQ6Samples, 0.5780322588980198);
}

// ---------------------------------------------------------------------------
// IQ2_XXS / IQ3_XXS / IQ2_S: block bytes come from a fixed LCG (seed 12345+t,
// fp16 scales 1.0 / 0.1 / -0.2) and the expected floats from llama.cpp commit
// 23f7aab dequantize_row_iq2_xxs / _iq3_xxs / _iq2_s run on those same bytes.
// Bit patterns of the first 16 outputs are checked directly; all 3*256 outputs
// are checked via a per-block FNV-1a 64 over the little-endian bit patterns.
// ---------------------------------------------------------------------------

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t Fnv1a(const float* values, std::size_t count) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t bits = FloatBits(values[i]);
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= (bits >> (8 * byte)) & 0xFFU;
      hash *= 0x100000001b3ULL;
    }
  }
  return hash;
}

void CheckLlamaOracle(gufo::core::GgmlType type, std::size_t block_bytes,
                      std::span<const std::uint8_t> bytes,
                      std::span<const std::uint32_t> first_bits,
                      std::span<const std::uint64_t> block_hashes,
                      void (*dequant)(const void*, float*, std::size_t)) {
  Expect(!gufo::core::ToString(type).empty(), "type has a name");
  Expect(gufo::quant::QuantizedBlockElements(type) == kBlockElements,
         "block elements");
  Expect(gufo::quant::EncodedSizeBytes(type, kBlockElements) == block_bytes,
         "one block encoded size");
  Expect(gufo::quant::EncodedSizeBytes(type, 4096) == 16 * block_bytes,
         "4096 elements encoded size");
  Expect(gufo::quant::EncodedSizeBytes(type, 100) == 0, "ragged row rejected");
  const std::size_t blocks = bytes.size() / block_bytes;
  Expect(blocks == block_hashes.size(), "block count");
  std::vector<float> out(blocks * kBlockElements, -777.0F);
  dequant(bytes.data(), out.data(), out.size());
  for (std::size_t i = 0; i < first_bits.size(); ++i) {
    if (FloatBits(out[i]) != first_bits[i]) {
      std::cerr << "Assertion failed: bit mismatch at " << i << " actual=0x"
                << std::hex << FloatBits(out[i]) << " expected=0x"
                << first_bits[i] << std::dec << '\n';
      std::exit(1);
    }
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    if (Fnv1a(out.data() + (b * kBlockElements), kBlockElements) !=
        block_hashes[b]) {
      std::cerr << "Assertion failed: block hash mismatch, block " << b << '\n';
      std::exit(1);
    }
  }
}

void TestIq2Xxs() {
  static constexpr std::uint8_t kBytes[198] = {
      0,   60,  139, 162, 232, 28,  126, 140, 152, 200, 10,  190, 247, 18,  179,
      117, 101, 245, 103, 243, 254, 169, 108, 127, 73,  108, 172, 39,  22,  218,
      79,  1,   118, 74,  146, 246, 3,   199, 77,  82,  244, 139, 170, 168, 159,
      41,  237, 196, 117, 185, 84,  182, 147, 194, 113, 52,  116, 171, 133, 223,
      100, 236, 190, 36,  175, 89,  102, 46,  203, 45,  30,  103, 223, 185, 137,
      63,  100, 222, 208, 95,  213, 177, 135, 229, 161, 235, 98,  200, 211, 24,
      158, 156, 28,  148, 227, 23,  70,  199, 187, 96,  46,  72,  225, 100, 198,
      183, 23,  188, 215, 143, 54,  231, 102, 52,  173, 210, 255, 0,   168, 5,
      90,  27,  130, 233, 252, 77,  86,  238, 231, 47,  185, 154, 102, 178, 250,
      197, 167, 115, 122, 125, 95,  90,  241, 98,  29,  160, 171, 114, 217, 36,
      33,  154, 147, 163, 115, 117, 147, 92,  162, 31,  79,  47,  148, 0,   25,
      223, 63,  234, 23,  90,  141, 1,   54,  37,  197, 56,  3,   84,  151, 97,
      173, 185, 30,  24,  155, 32,  100, 19,  69,  197, 68,  134, 81,  105, 187,
      190, 32,  165,
  };
  static constexpr std::uint32_t kFirstBits[16] = {
      0x41c80000U, 0xc3066000U, 0xc1c80000U, 0xc29c4000U,
      0xc1c80000U, 0xc1c80000U, 0xc1c80000U, 0x429c4000U,
      0x41c80000U, 0x41c80000U, 0x41c80000U, 0xc1c80000U,
      0xc3066000U, 0x429c4000U, 0x41c80000U, 0x429c4000U};
  static constexpr std::uint64_t kHashes[3] = {
      0xae423ba7e3674d19ULL, 0x1fbdb10cd750f1dbULL, 0xcf07e1de1b299838ULL};
  CheckLlamaOracle(gufo::core::GgmlType::kIQ2_XXS, 66, kBytes, kFirstBits,
                   kHashes, &gufo::quant::DequantizeIQ2_XXS);
}

void TestIq3Xxs() {
  static constexpr std::uint8_t kBytes[294] = {
      0,   60,  58,  172, 147, 220, 237, 118, 26,  215, 126, 109, 241, 6,   237,
      237, 140, 249, 180, 214, 175, 57,  23,  146, 103, 221, 70,  236, 230, 120,
      230, 41,  109, 229, 248, 226, 223, 154, 51,  119, 175, 111, 58,  54,  58,
      41,  83,  130, 193, 20,  134, 146, 160, 139, 60,  157, 108, 60,  211, 132,
      9,   123, 189, 76,  2,   2,   21,  5,   145, 216, 203, 235, 58,  99,  205,
      208, 219, 185, 210, 51,  152, 197, 167, 134, 5,   220, 36,  133, 110, 197,
      48,  167, 47,  95,  113, 9,   103, 93,  102, 46,  142, 204, 74,  160, 164,
      85,  105, 251, 16,  113, 192, 154, 101, 4,   155, 143, 147, 12,  9,   80,
      76,  101, 82,  240, 164, 198, 57,  119, 35,  122, 248, 213, 219, 125, 12,
      88,  62,  251, 70,  173, 186, 93,  186, 200, 98,  92,  106, 65,  34,  205,
      231, 251, 70,  29,  49,  45,  91,  168, 148, 30,  160, 110, 76,  38,  164,
      200, 45,  202, 186, 33,  158, 14,  91,  223, 158, 64,  169, 237, 159, 224,
      175, 165, 124, 119, 3,   0,   202, 98,  232, 79,  65,  252, 43,  226, 30,
      170, 102, 178, 144, 161, 49,  165, 176, 124, 27,  171, 142, 249, 61,  107,
      74,  104, 92,  238, 78,  168, 138, 57,  28,  197, 134, 91,  71,  131, 247,
      20,  126, 119, 6,   189, 219, 123, 29,  121, 189, 135, 199, 204, 243, 94,
      0,   30,  254, 132, 108, 214, 165, 103, 74,  3,   46,  193, 20,  191, 237,
      151, 27,  214, 3,   84,  169, 69,  119, 233, 93,  165, 13,  244, 206, 157,
      118, 79,  187, 224, 210, 152, 73,  96,  139, 127, 20,  175, 6,   245, 17,
      194, 194, 146, 144, 141, 198, 189, 219, 30,
  };
  static constexpr std::uint32_t kFirstBits[16] = {
      0xc2a20000U, 0x41d80000U, 0xc1d80000U, 0x42a20000U,
      0xc33d0000U, 0x433d0000U, 0x43d14000U, 0xc33d0000U,
      0x43070000U, 0xc3d14000U, 0x41d80000U, 0xc33d0000U,
      0x43730000U, 0x43730000U, 0x41d80000U, 0x43af8000U};
  static constexpr std::uint64_t kHashes[3] = {
      0xcede3e7211f68c28ULL, 0xd006d9a89330198cULL, 0x1ec73a29156fe7feULL};
  CheckLlamaOracle(gufo::core::GgmlType::kIQ3_XXS, 98, kBytes, kFirstBits,
                   kHashes, &gufo::quant::DequantizeIQ3_XXS);
}

void TestIq2S() {
  static constexpr std::uint8_t kBytes[246] = {
      0,   60,  233, 181, 62,  155, 91,  97,  157, 229, 242, 29,  235, 250, 40,
      100, 179, 253, 0,   185, 96,  202, 195, 164, 133, 78,  223, 176, 181, 22,
      124, 81,  99,  129, 94,  206, 186, 108, 25,  157, 107, 83,  201, 197, 212,
      40,  185, 64,  13,  111, 183, 110, 173, 83,  7,   7,   99,  205, 33,  41,
      175, 11,  189, 116, 85,  170, 207, 64,  87,  132, 119, 111, 148, 13,  16,
      97,  82,  148, 212, 7,   91,  217, 102, 46,  105, 205, 231, 65,  9,   114,
      195, 178, 66,  43,  255, 252, 137, 244, 108, 82,  238, 81,  179, 219, 130,
      242, 188, 58,  74,  83,  75,  78,  101, 211, 137, 77,  38,  24,  107, 154,
      62,  175, 34,  247, 76,  63,  29,  0,   96,  198, 55,  17,  75,  163, 29,
      235, 212, 132, 19,  221, 20,  96,  131, 47,  166, 24,  41,  16,  106, 119,
      172, 92,  250, 151, 239, 229, 34,  244, 134, 29,  240, 173, 4,   75,  102,
      178, 26,  171, 93,  232, 175, 27,  127, 153, 118, 72,  80,  134, 168, 96,
      177, 145, 219, 213, 106, 225, 48,  177, 139, 218, 63,  114, 4,   90,  128,
      151, 76,  18,  35,  42,  36,  231, 130, 10,  204, 192, 178, 185, 70,  33,
      216, 245, 104, 113, 56,  89,  239, 234, 61,  208, 194, 100, 222, 162, 25,
      163, 40,  75,  77,  197, 195, 127, 100, 241, 205, 1,   177, 124, 89,  42,
      31,  14,  99,  71,  242, 136,
  };
  static constexpr std::uint32_t kFirstBits[16] = {
      0x40ac0000U, 0xc0ac0000U, 0xbf800000U, 0xc0ac0000U,
      0xc0ac0000U, 0x3f800000U, 0xc0ac0000U, 0x40ac0000U,
      0x40ac0000U, 0xc0480000U, 0xc0ac0000U, 0xc0ac0000U,
      0x40ac0000U, 0x3f800000U, 0xc0480000U, 0xc0ac0000U};
  static constexpr std::uint64_t kHashes[3] = {
      0x9bccd9053fa72ea8ULL, 0xa0712b4af725c46bULL, 0x4653281decfd1648ULL};
  CheckLlamaOracle(gufo::core::GgmlType::kIQ2_S, 82, kBytes, kFirstBits,
                   kHashes, &gufo::quant::DequantizeIQ2_S);
}

}  // namespace

int main() {
  TestQ4();
  TestQ6();
  TestQ3();
  TestQ8();
  TestQ8_0();
  TestQ5_K();
  TestIq2Xxs();
  TestIq3Xxs();
  TestIq2S();
  if (const char* model = std::getenv("GUFO_MTP_MODEL");
      model != nullptr && std::string_view(model).size() > 0) {
    TestModelRows(model);
  }
  return 0;
}
