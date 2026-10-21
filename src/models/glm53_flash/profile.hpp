#ifndef GUFO_MODELS_GLM53_FLASH_PROFILE_HPP_
#define GUFO_MODELS_GLM53_FLASH_PROFILE_HPP_

// Per-op GPU timer for the GLM-5.3-Flash layer loop (GUFO_GLM53_PROFILE).
//
// GUFO_GLM53_PROFILE=1 makes the Executor bracket each op with a pair of
// hipEvents on its own stream, resolve the elapsed times after the pass's
// existing end-of-pass synchronisation, and aggregate them per op name and
// phase (prefill = a pass of more than one token, decode = one token). The
// table goes to stderr when the Executor is destroyed; "2" also prints it
// after every Forward. Unset, empty or "0": no profiler object exists, no
// events are created and no synchronisation is added (an OpScope on a null
// profiler is a pointer test).
//
// Event pairs are enqueued in stream order and are non-blocking, so the
// numbers are the stream time of each op with no tap-style synchronisation
// in between (the events themselves add a small per-op gap, about a
// microsecond, so ops under that size read high). Nested ops use a '/' in
// the name ("mla/indexer" is inside "mla"); shares are of the top-level
// ops, whose sum is the pass total less unaccounted gaps.

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace gufo::models::glm53_flash {

class OpProfiler {
  struct Span {
    const char* name{nullptr};
    hipStream_t stream{};
    hipEvent_t start{};
    hipEvent_t stop{};
  };
  struct Stat {
    std::uint64_t calls{0};
    double ms{0.0};
  };
  struct Phase {
    std::uint64_t passes{0};
    std::uint64_t tokens{0};
    std::map<std::string, Stat> ops;
  };

public:
  /// nullptr unless GUFO_GLM53_PROFILE is set to something other than "" / "0".
  [[nodiscard]] static std::unique_ptr<OpProfiler> FromEnv() {
    const char* v = std::getenv("GUFO_GLM53_PROFILE");
    if (v == nullptr || *v == '\0' || std::strcmp(v, "0") == 0) {
      return nullptr;
    }
    return std::unique_ptr<OpProfiler>(
        new OpProfiler(std::strcmp(v, "2") == 0));
  }

  ~OpProfiler() {
    for (const hipEvent_t e : free_) {
      (void)hipEventDestroy(e);
    }
    for (const Span& s : open_) {
      (void)hipEventDestroy(s.start);
      (void)hipEventDestroy(s.stop);
    }
  }
  OpProfiler(const OpProfiler&) = delete;
  OpProfiler& operator=(const OpProfiler&) = delete;

  /// True if the table should also be printed after every Forward.
  [[nodiscard]] bool per_run() const noexcept { return per_run_; }

  /// Starts a pass; `tokens` selects the phase.
  void BeginPass(std::uint32_t tokens) {
    phase_ = tokens > 1 ? 0 : 1;
    pass_tokens_ = tokens;
  }

  /// Resolves the spans recorded since BeginPass. Call after the stream has
  /// been synchronised past every span's stop event.
  void EndPass() {
    Phase& ph = phases_[phase_];
    ++ph.passes;
    ph.tokens += pass_tokens_;
    for (const Span& s : open_) {
      float ms = 0.0F;
      if (hipEventElapsedTime(&ms, s.start, s.stop) == hipSuccess) {
        Stat& st = ph.ops[s.name];
        ++st.calls;
        st.ms += ms;
      }
      free_.push_back(s.start);
      free_.push_back(s.stop);
    }
    open_.clear();
  }

  /// Drops spans of a pass that failed before its synchronisation.
  void AbortPass() {
    for (const Span& s : open_) {
      free_.push_back(s.start);
      free_.push_back(s.stop);
    }
    open_.clear();
  }

  void Print(std::ostream& os) const {
    static constexpr const char* kPhaseName[2] = {"prefill passes",
                                                  "decode passes"};
    for (int p = 0; p < 2; ++p) {
      const Phase& ph = phases_[p];
      if (ph.passes == 0) {
        continue;
      }
      double top_total = 0.0;
      for (const auto& [name, st] : ph.ops) {
        if (name.find('/') == std::string::npos && name != "pass") {
          top_total += st.ms;
        }
      }
      const auto pass_it = ph.ops.find("pass");
      const double pass_ms = pass_it != ph.ops.end() ? pass_it->second.ms : 0.0;
      char line[256];
      std::snprintf(line, sizeof line,
                    "[glm53 profile] %s: %llu passes, %llu tokens, %.1f ms "
                    "total (%.2f ms/pass)\n",
                    kPhaseName[p], static_cast<unsigned long long>(ph.passes),
                    static_cast<unsigned long long>(ph.tokens), pass_ms,
                    pass_ms / static_cast<double>(ph.passes));
      os << line;
      std::snprintf(line, sizeof line,
                    "[glm53 profile]   %-24s %9s %11s %11s %7s\n", "op",
                    "calls", "total ms", "ms/call", "share");
      os << line;
      std::vector<std::pair<std::string, Stat>> rows(ph.ops.begin(),
                                                     ph.ops.end());
      std::stable_sort(rows.begin(), rows.end(),
                       [](const auto& a, const auto& b) {
                         return a.second.ms > b.second.ms;
                       });
      for (const auto& [name, st] : rows) {
        if (name == "pass") {
          continue;
        }
        const bool nested = name.find('/') != std::string::npos;
        char share[16];
        if (nested || top_total <= 0.0) {
          std::snprintf(share, sizeof share, "%s", "  (in)");
        } else {
          std::snprintf(share, sizeof share, "%5.1f%%",
                        100.0 * st.ms / top_total);
        }
        std::snprintf(line, sizeof line,
                      "[glm53 profile]   %-24s %9llu %11.2f %11.4f %7s\n",
                      name.c_str(), static_cast<unsigned long long>(st.calls),
                      st.ms, st.ms / static_cast<double>(st.calls), share);
        os << line;
      }
      std::snprintf(line, sizeof line,
                    "[glm53 profile]   top-level ops sum %.1f ms; pass total "
                    "%.1f ms\n",
                    top_total, pass_ms);
      os << line;
    }
  }

  /// RAII event pair. `name` must be a string literal / outlive the pass.
  class Scope {
  public:
    Scope(OpProfiler* p, const char* name, hipStream_t stream) : p_(p) {
      if (p_ != nullptr) {
        span_.name = name;
        span_.stream = stream;
        if (!p_->Take(&span_.start) || !p_->Take(&span_.stop) ||
            hipEventRecord(span_.start, stream) != hipSuccess) {
          p_ = nullptr;
        }
      }
    }
    ~Scope() {
      if (p_ != nullptr) {
        if (hipEventRecord(span_.stop, span_.stream) == hipSuccess) {
          p_->open_.push_back(span_);
        } else {
          p_->free_.push_back(span_.start);
          p_->free_.push_back(span_.stop);
        }
      }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

  private:
    OpProfiler* p_;
    Span span_{};
  };

private:
  explicit OpProfiler(bool per_run) : per_run_(per_run) {}

  bool Take(hipEvent_t* e) {
    if (!free_.empty()) {
      *e = free_.back();
      free_.pop_back();
      return true;
    }
    return hipEventCreate(e) == hipSuccess;
  }

  bool per_run_;
  int phase_{0};
  std::uint32_t pass_tokens_{0};
  Phase phases_[2];
  std::vector<Span> open_;
  std::vector<hipEvent_t> free_;
};

/// Times the enclosing scope on `stream`; free when `profiler` is null.
using OpScope = OpProfiler::Scope;

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_PROFILE_HPP_
