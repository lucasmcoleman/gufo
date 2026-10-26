// `gufo bench` on Laguna: the prefill (pp), decode (tg) and depth rows and the
// prefill validation, on a tiny random model through the same function the
// real GGUF reaches (RunLagunaBenchmarkOnModel). No model file; nothing is
// skipped. The table's rows are checked, not the speeds.
#include "src/cli/bench/bench.hpp"

#include <iostream>
#include <sstream>
#include <string>

#include "tests/models/laguna/engine_fixture.hpp"

using namespace gufo::testing::laguna;

namespace {

std::string RunCaptured(
    const gufo::cli::BenchOptions& options,
    const std::shared_ptr<gufo::models::laguna::Model>& model, int* status) {
  std::ostringstream out;
  auto* previous = std::cout.rdbuf(out.rdbuf());
  *status = gufo::cli::RunLagunaBenchmarkOnModel(options, model, 1.0);
  std::cout.rdbuf(previous);
  return out.str();
}

void Expect(const std::string& table, const std::string& needle) {
  Require(table.find(needle) != std::string::npos,
          "the bench table lacks '" + needle + "':\n" + table);
}

}  // namespace

int main() {
  try {
    const auto fixture = MakeEngineFixture();
    gufo::cli::BenchOptions options;
    options.n_prompts = {64, 100};
    options.n_gens = {8};
    options.n_depths = {0, 48};
    options.repetitions = 2;
    int status = 0;
    const std::string table = RunCaptured(options, fixture.model, &status);
    Require(status == 0, "bench exit status:\n" + table);
    for (const char* row :
         {" pp64 |", " pp100 |", " tg8 |", " pp64 @ d48 |", " pp100 @ d48 |",
          " tg8 @ d48 |", "Laguna-tiny", "ROCm (HIP)"}) {
      Expect(table, row);
    }
    Require(table.find("nan") == std::string::npos &&
                table.find("inf") == std::string::npos,
            "the table has no nan or inf:\n" + table);

    // A workload beyond the model's context is refused before any timing.
    options.n_prompts = {600};
    options.n_depths = {0};
    const std::string refused = RunCaptured(options, fixture.model, &status);
    Require(status != 0 && refused.find(" pp600 |") == std::string::npos,
            "an over-long prompt is refused");

    // Prefill validation prints its comparison (the winner may differ on a
    // random model, so only the report and finiteness are required here).
    options.n_prompts = {32};
    options.n_gens.clear();
    options.validate_prefill_tokens = 40;
    const std::string validated = RunCaptured(options, fixture.model, &status);
    Expect(validated, "[Prefill Validation] tokens=40 finite=yes");
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "laguna.bench: ok\n";
  return 0;
}
