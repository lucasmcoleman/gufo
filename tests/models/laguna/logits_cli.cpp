// laguna_logits: teacher-forced all-position logit dump for the Laguna-S-2.1
// GPU executor, implementing the codexms #969 contract
// (/server/staging/laguna/codexms-969-README.txt).
//
//   laguna_logits --reference DIR --output DIR [--model GGUF] [--ubatch N]
//                 [--limit N] [--context N] [--budget-only]
//
// DIR/tokens.i32 ([S,T] little-endian int32) and DIR/index.json come from the
// reference run (llama_logits). Every sequence runs from position 0 on a
// cleared cache in chunks of --ubatch tokens (default 256), requesting the
// logits of all T rows, and is written as NN.f32 ([T,n_vocab] float32). The
// output directory also gets its own tokens.i32 (the tokens actually executed)
// and an index.json (complete=true, n_vocab, T, ordered seqs with file,
// n_tokens, tail, logits) that is published, by rename, only after every other
// file is written. Nonfinite logits fail the run; a `.incomplete` marker stays
// until the manifest is published. The output directory must be new or empty.
//
// --limit N runs only the first N sequences (a pilot; score.py compares whole
// manifests, so score a pilot with the pilot snippet, not score.py).
//
// The memory budget (weights, KV at the context, scratch, logits) is printed
// before anything is uploaded. --budget-only prints it and exits without
// touching the GPU. If the budget does not fit the free device memory the run
// stops before the first upload. The 74 GB model does not fit beside another
// resident model: coordinate the GPU window first.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

#include "src/core/hip/heap_placement.hpp"
#include "src/models/laguna/budget.hpp"
#include "src/models/laguna/config.hpp"
#include "src/models/laguna/kernels/device_model.hpp"
#include "src/models/laguna/kernels/executor.hpp"
#include "src/models/laguna/weights.hpp"
#include "tests/models/laguna/logits_dump.hpp"
#include "tests/models/laguna/support.hpp"

namespace ml = gufo::models::laguna;
namespace gpu = gufo::models::laguna::rocm;
using namespace gufo::testing::laguna;

namespace {

struct Args {
  fs::path model;
  fs::path reference;
  fs::path output;
  std::uint32_t ubatch{256};
  std::uint32_t limit{0};    // 0: all
  std::uint32_t context{0};  // 0: T
  bool budget_only{false};
};

std::uint32_t ParseUint(const std::string& flag, const std::string& text) {
  std::uint32_t value = 0;
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size() || value == 0) {
    DumpFail(flag + " needs a positive integer, got '" + text + "'");
  }
  return value;
}

Args Parse(int argc, char** argv) {
  Args a;
  a.model = ModelPath();
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        DumpFail(flag + " needs a value");
      }
      return argv[++i];
    };
    if (flag == "--model") {
      a.model = value();
    } else if (flag == "--reference") {
      a.reference = value();
    } else if (flag == "--output") {
      a.output = value();
    } else if (flag == "--ubatch") {
      a.ubatch = ParseUint(flag, value());
    } else if (flag == "--limit") {
      a.limit = ParseUint(flag, value());
    } else if (flag == "--context") {
      a.context = ParseUint(flag, value());
    } else if (flag == "--budget-only") {
      a.budget_only = true;
    } else {
      DumpFail("unknown option " + flag +
               "\nusage: laguna_logits --reference DIR --output DIR [--model "
               "GGUF] [--ubatch N] [--limit N] [--context N] [--budget-only]");
    }
  }
  if (a.reference.empty()) {
    DumpFail("--reference DIR is required");
  }
  if (a.output.empty() && !a.budget_only) {
    DumpFail("--output DIR is required");
  }
  return a;
}

int Run(int argc, char** argv) {
  const Args args = Parse(argc, argv);
  const Reference ref = ReadReference(args.reference);
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(args.model, &error);
  if (reader == nullptr) {
    DumpFail("cannot open " + args.model.string() + ": " + error);
  }
  auto config = ml::Config::FromGguf(*reader, &error);
  if (!config) {
    DumpFail("config: " + error);
  }
  auto weights = ml::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    DumpFail("weights: " + error);
  }
  if (config->vocab_size != ref.n_vocab) {
    DumpFail("reference n_vocab " + std::to_string(ref.n_vocab) +
             " differs from the model's " + std::to_string(config->vocab_size));
  }
  const std::uint32_t context = std::max(args.context, ref.T);
  const std::uint32_t ubatch = std::min(args.ubatch, context);
  const ml::MemoryBudget budget =
      ml::ComputeBudget(*weights, {context, ubatch, true});
  std::cout << budget.Format() << std::flush;
  if (args.budget_only) {
    std::cout << "budget only: no GPU memory was touched\n";
    return 0;
  }

  // Refuse before the first upload when the budget cannot fit.
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
    DumpFail("hipMemGetInfo failed");
  }
  if (!gufo::hip::HeapFitsAboveReserve(free_bytes, budget.device_total())) {
    DumpFail(
        "the budget does not fit: " + std::to_string(budget.device_total()) +
        " bytes needed above a " + std::to_string(budget.reserve) +
        " byte reserve, " + std::to_string(free_bytes) +
        " bytes free on the device. Another model is probably resident; "
        "coordinate the GPU window or lower --context/--ubatch.");
  }
  if (fs::exists(args.output) && !fs::is_empty(args.output)) {
    DumpFail(args.output.string() + " must be a new or empty directory");
  }

  const auto load_start = std::chrono::steady_clock::now();
  auto model = gpu::DeviceModel::Upload(*weights, reader.get(), &error);
  if (model == nullptr) {
    DumpFail("upload failed: " + error);
  }
  gpu::ExecutorOptions options;
  options.context = context;
  options.max_chunk = ubatch;
  auto ex = gpu::Executor::Create(*model, options, &error);
  if (ex == nullptr) {
    DumpFail("executor: " + error);
  }
  const double load_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - load_start)
                            .count();
  std::cout << "loaded in " << load_s << " s: weights "
            << model->resident_bytes() << " B, kv " << ex->kv_bytes()
            << " B, warm-up measured " << ex->measured_warmup_bytes()
            << " B (estimate " << budget.scratch() << " B)\n"
            << std::flush;

  DumpOptions dump;
  dump.output = args.output;
  dump.reference = args.reference;
  dump.model = args.model.string();
  dump.ubatch = ubatch;
  dump.context = context;
  dump.sequences = args.limit;
  DumpLogits(ref, *ex, dump, &std::cerr);
  std::cout << "wrote "
            << (args.limit == 0
                    ? ref.seqs.size()
                    : std::min<std::size_t>(args.limit, ref.seqs.size()))
            << " sequences to " << args.output.string() << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "laguna_logits: " << e.what() << "\n";
    return 1;
  }
}
