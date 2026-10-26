// Laguna-S-2.1 GPU executor. See executor.hpp.
//
// The RMSNorm and residual kernels use plain IEEE arithmetic (no fast-math, no
// FP contraction) and accumulate the norm's sum in double, as the CPU oracle
// does, so a row's result depends only on the row.
#pragma clang fp contract(off)

#include "src/models/laguna/kernels/executor.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "src/core/hip/heap_placement.hpp"
#include "src/models/laguna/budget.hpp"
#include "src/models/laguna/reference.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::laguna::rocm {
namespace {

constexpr int kNormThreads = 256;
constexpr std::uint32_t kVectorMaxRows = 32;

void Assign(std::string* out, const std::string& message) {
  if (out != nullptr) {
    *out = message;
  }
}

hipError_t DefaultAllocate(void** ptr, std::size_t bytes) {
  return hipMalloc(ptr, bytes);
}
hipError_t DefaultRelease(void* ptr) {
  return hipFree(ptr);
}

/// y[r] = (x[r] * scale) * w, scale = 1 / sqrt(mean(x^2) + eps): the oracle's
/// RmsNormRow. One block per row.
__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ y, std::uint32_t n,
                              float eps) {
  const float* xr = x + std::size_t{blockIdx.x} * n;
  float* yr = y + std::size_t{blockIdx.x} * n;
  double sum = 0.0;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    sum += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
  }
  __shared__ double part[kNormThreads];
  part[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      part[threadIdx.x] += part[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = static_cast<float>(part[0] / static_cast<double>(n));
  const float scale = 1.0F / sqrtf(mean + eps);
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    yr[i] = (xr[i] * scale) * w[i];
  }
}

__global__ void AddKernel(const float* __restrict__ a,
                          const float* __restrict__ b, float* __restrict__ out,
                          std::size_t count) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = a[i] + b[i];
  }
}

}  // namespace

Executor::~Executor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  kv_.clear();
  attn_.reset();
  ffn_.reset();
  for (Buffer* b : {&h_, &n_, &x_, &o_, &logits_, &zero_ids_, &rows_}) {
    Free(b);
  }
  if (stream_ != nullptr) {
    qfn_mmq_release_stream(stream_);
    (void)hipStreamDestroy(stream_);
  }
}

bool Executor::Alloc(Buffer* b, std::size_t bytes, std::string* error_msg) {
  void* ptr = nullptr;
  if (const hipError_t s = allocator_.allocate(&ptr, bytes); s != hipSuccess) {
    Assign(error_msg, "laguna executor: allocating " + std::to_string(bytes) +
                          " bytes failed: " + hipGetErrorString(s));
    return false;
  }
  b->ptr = ptr;
  b->bytes = bytes;
  return true;
}

void Executor::Free(Buffer* b) noexcept {
  if (b->ptr != nullptr) {
    (void)allocator_.release(b->ptr);
  }
  b->ptr = nullptr;
  b->bytes = 0;
}

bool Executor::Fail(std::string* error_msg, const std::string& message) {
  poisoned_ = true;
  Assign(error_msg, message);
  return false;
}

std::size_t Executor::kv_bytes() const noexcept {
  std::size_t total = 0;
  for (const KvCache& cache : kv_) {
    total += cache.device_bytes();
  }
  return total;
}

std::size_t Executor::scratch_bytes() const noexcept {
  return h_.bytes + n_.bytes + x_.bytes + o_.bytes + logits_.bytes +
         zero_ids_.bytes + rows_.bytes +
         (ffn_ != nullptr ? ffn_->scratch_bytes() : 0);
}

bool Executor::BuildCaches(std::string* error_msg) {
  const Config& config = model_->config();
  kv_.clear();
  kv_.reserve(config.num_layers);
  for (std::uint32_t il = 0; il < config.num_layers; ++il) {
    auto cache = KvCache::Create(config, il, options_.context,
                                 options_.max_chunk, error_msg);
    if (!cache) {
      kv_.clear();
      return false;
    }
    kv_.push_back(std::move(*cache));
  }
  return true;
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           const ExecutorOptions& options,
                                           std::string* error_msg) {
  const Config& config = model.config();
  if (options.context == 0 || options.max_chunk == 0 ||
      (options.path == Path::kVector && options.max_chunk > kVectorMaxRows)) {
    Assign(error_msg, "laguna executor: bad context or chunk size");
    return nullptr;
  }
  // The weights are already resident, so what must fit is everything else.
  const MemoryBudget budget = ComputeBudget(
      model.weights(), BudgetRequest{options.context, options.max_chunk, true});
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
    const std::size_t need = budget.kv() + budget.scratch();
    if (!hip::HeapFitsAboveReserve(free_bytes, need)) {
      Assign(error_msg,
             "laguna executor: KV and scratch need " + std::to_string(need) +
                 " bytes but the device has " + std::to_string(free_bytes) +
                 " free (reserve " + std::to_string(budget.reserve) +
                 "); lower the context or the chunk size\n" + budget.Format());
      return nullptr;
    }
  }
  std::unique_ptr<Executor> ex(new Executor(model, options));
  ex->allocator_ = options.allocator;
  if (ex->allocator_.allocate == nullptr) {
    ex->allocator_.allocate = DefaultAllocate;
  }
  if (ex->allocator_.release == nullptr) {
    ex->allocator_.release = DefaultRelease;
  }
  if (const hipError_t s =
          hipStreamCreateWithFlags(&ex->stream_, hipStreamNonBlocking);
      s != hipSuccess) {
    Assign(error_msg,
           std::string("laguna executor: stream: ") + hipGetErrorString(s));
    return nullptr;
  }
  ex->attn_ = std::make_unique<AttnOps>(ex->stream_);
  ex->ffn_ = std::make_unique<FfnOps>(ex->stream_);
  ex->attn_->SetAllocator(ex->allocator_);
  ex->ffn_->SetAllocatorForTesting(ex->allocator_.allocate,
                                   ex->allocator_.release);
  const std::size_t C = options.max_chunk;
  const std::size_t H = config.hidden_size;
  const std::size_t rows = C * H * sizeof(float);
  if (!ex->Alloc(&ex->h_, rows, error_msg) ||
      !ex->Alloc(&ex->n_, rows, error_msg) ||
      !ex->Alloc(&ex->x_, rows, error_msg) ||
      !ex->Alloc(&ex->o_, rows, error_msg) ||
      !ex->Alloc(&ex->logits_, C * config.vocab_size * sizeof(float),
                 error_msg) ||
      !ex->Alloc(&ex->zero_ids_, C * sizeof(std::int32_t), error_msg)) {
    return nullptr;
  }
  if (hipMemset(ex->zero_ids_.ptr, 0, ex->zero_ids_.bytes) != hipSuccess) {
    Assign(error_msg, "laguna executor: clearing the id row failed");
    return nullptr;
  }
  if (!ex->BuildCaches(error_msg)) {
    return nullptr;
  }
  if (options.warmup) {
    std::size_t before = 0;
    std::size_t after = 0;
    std::size_t total = 0;
    const bool have_info = hipMemGetInfo(&before, &total) == hipSuccess;
    if (hipMemset(ex->h_.ptr, 0, ex->h_.bytes) != hipSuccess ||
        !ex->Run(options.max_chunk, 0, LogitsRows::kAll, error_msg)) {
      return nullptr;
    }
    if (have_info && hipMemGetInfo(&after, &total) == hipSuccess &&
        before > after) {
      ex->measured_warmup_ = before - after;
    }
    if (!ex->BuildCaches(error_msg)) {
      return nullptr;
    }
  }
  return ex;
}

bool Executor::Reset(std::string* error_msg) {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  position_ = 0;
  if (!BuildCaches(error_msg)) {
    poisoned_ = true;
    return false;
  }
  poisoned_ = false;
  return true;
}

bool Executor::Clear(std::string* error_msg) {
  if (poisoned_) {
    return Reset(error_msg);
  }
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  for (KvCache& cache : kv_) {
    cache.Reset();
  }
  position_ = 0;
  return true;
}

std::size_t Executor::snapshot_bytes() const noexcept {
  std::size_t total = 0;
  for (const KvCache& cache : kv_) {
    total += std::size_t{cache.committed() - cache.retained_begin()} *
             cache.row_bytes() * 2;
  }
  return total;
}

bool Executor::SaveSnapshot(ExecutorSnapshot* snapshot,
                            std::string* error_msg) {
  if (poisoned_) {
    Assign(error_msg,
           "laguna executor: a previous call failed; Reset() is required");
    return false;
  }
  ExecutorSnapshot out;
  out.position = position_;
  out.layers.resize(kv_.size());
  for (std::size_t il = 0; il < kv_.size(); ++il) {
    if (kv_[il].committed() != position_) {
      Assign(error_msg, "laguna executor: layer " + std::to_string(il) +
                            " is not at the session position");
      return false;
    }
    if (!kv_[il].Export(stream_, &out.layers[il], error_msg)) {
      return false;
    }
  }
  *snapshot = std::move(out);
  return true;
}

bool Executor::RestoreSnapshot(const ExecutorSnapshot& snapshot,
                               std::string* error_msg) {
  if (poisoned_) {
    Assign(error_msg,
           "laguna executor: a previous call failed; Reset() is required");
    return false;
  }
  if (snapshot.layers.size() != kv_.size() ||
      snapshot.position > options_.context) {
    Assign(error_msg, "laguna executor: snapshot does not fit this executor");
    return false;
  }
  for (const KvImage& layer : snapshot.layers) {
    if (layer.committed != snapshot.position) {
      Assign(error_msg, "laguna executor: snapshot layers disagree");
      return false;
    }
  }
  for (std::size_t il = 0; il < kv_.size(); ++il) {
    if (!kv_[il].Import(stream_, snapshot.layers[il], error_msg)) {
      // Earlier layers already hold the snapshot and this one is invalid.
      poisoned_ = true;
      position_ = 0;
      return false;
    }
  }
  position_ = snapshot.position;
  return true;
}

bool Executor::Run(std::uint32_t T, std::uint32_t start, LogitsRows rows,
                   std::string* error_msg, std::span<Executor* const> batch) {
  const Config& config = model_->config();
  const std::uint32_t H = config.hidden_size;
  Path path = options_.path;
  if (path == Path::kAuto) {
    path = T <= kVectorMaxTokens ? Path::kVector : Path::kBatched;
  }
  if (!batch.empty()) {
    // Decode rows of several sessions always take the vector kernels, the
    // path a lone decode token takes.
    path = Path::kVector;
  }
  std::vector<KvCache*> layer_caches(batch.size());
  const bool vector = path == Path::kVector;
  float* h = static_cast<float*>(h_.ptr);
  float* n = static_cast<float*>(n_.ptr);
  float* x = static_cast<float*>(x_.ptr);
  float* o = static_cast<float*>(o_.ptr);
  const auto check = [&](const char* what) {
    const hipError_t s = hipGetLastError();
    return s == hipSuccess
               ? true
               : Fail(error_msg, std::string("laguna executor: ") + what +
                                     ": " + hipGetErrorString(s));
  };
  const auto norm = [&](const float* in, const float* w, float* out,
                        std::uint32_t count) {
    RmsNormKernel<<<count, kNormThreads, 0, stream_>>>(in, w, out, H,
                                                       config.rms_eps);
  };
  const auto add = [&](const float* a, const float* b, float* out) {
    const std::size_t count = std::size_t{T} * H;
    AddKernel<<<(count + 255) / 256, 256, 0, stream_>>>(a, b, out, count);
  };
  std::string op_error;
  for (std::uint32_t il = 0; il < config.num_layers; ++il) {
    const DeviceLayer& layer = model_->layers()[il];
    norm(h, layer.attn_norm, n, T);
    bool attended = false;
    if (check("attention norm")) {
      if (batch.empty()) {
        attended = attn_->Forward(layer.attn, kv_[il], n, o, T, start, path,
                                  &op_error);
      } else {
        for (std::size_t t = 0; t < batch.size(); ++t) {
          layer_caches[t] = &batch[t]->kv_[il];
        }
        attended = attn_->ForwardBatch(
            layer.attn, layer_caches,
            static_cast<const DecodeRow*>(rows_.ptr) + std::size_t{il} * T, n,
            o, &op_error);
      }
    }
    if (!attended) {
      return Fail(error_msg,
                  !op_error.empty()
                      ? "laguna layer " + std::to_string(il) + ": " + op_error
                      : std::string());
    }
    add(o, h, x);
    norm(x, layer.ffn_norm, n, T);
    if (!check("ffn norm")) {
      return false;
    }
    const bool ok =
        layer.ffn.moe()
            ? ffn_->Moe(layer.ffn.routed(), n, o, H, T, path, &op_error)
            : ffn_->Dense(layer.ffn.dense(), n, o, T, path, &op_error);
    if (!ok) {
      return Fail(error_msg,
                  "laguna layer " + std::to_string(il) + ": " + op_error);
    }
    add(o, x, h);
    if (!check("residual")) {
      return false;
    }
    if (layer_tap_) {
      if (const hipError_t s = hipStreamSynchronize(stream_); s != hipSuccess) {
        return Fail(error_msg, std::string("laguna executor: tap sync: ") +
                                   hipGetErrorString(s));
      }
      layer_tap_(il, h, std::size_t{T} * H);
    }
  }
  if (rows != LogitsRows::kNone) {
    const std::uint32_t N = rows == LogitsRows::kAll ? T : 1;
    const float* in =
        h + (rows == LogitsRows::kAll ? 0 : std::size_t{T - 1} * H);
    norm(in, model_->output_norm(), n, N);
    if (!check("output norm")) {
      return false;
    }
    const DeviceMatrix& w = model_->output();
    const auto* ids = static_cast<const std::int32_t*>(zero_ids_.ptr);
    const int M = static_cast<int>(w.rows);
    const int K = static_cast<int>(w.cols);
    const int rc =
        vector ? qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, n, ids,
                                 static_cast<float*>(logits_.ptr), M, K,
                                 static_cast<int>(N), 1, 1, stream_)
               : qfn_mmq_q6_K_moe_raw(w.data, n, ids,
                                      static_cast<float*>(logits_.ptr), M, K,
                                      static_cast<int>(N), 1, 1, stream_);
    if (rc != 0) {
      return Fail(error_msg, "laguna executor: output projection failed");
    }
  }
  if (const hipError_t s = hipStreamSynchronize(stream_); s != hipSuccess) {
    return Fail(error_msg,
                std::string("laguna executor: ") + hipGetErrorString(s));
  }
  return true;
}

bool Executor::Forward(std::span<const std::int32_t> tokens, LogitsRows rows,
                       float* logits, std::string* error_msg) {
  if (poisoned_) {
    Assign(error_msg,
           "laguna executor: a previous call failed; Reset() is required");
    return false;
  }
  const Config& config = model_->config();
  const std::size_t H = config.hidden_size;
  const std::size_t T = tokens.size();
  if (T == 0 || T > options_.max_chunk) {
    Assign(error_msg, "laguna executor: chunk of " + std::to_string(T) +
                          " rows outside 1.." +
                          std::to_string(options_.max_chunk));
    return false;
  }
  if (std::uint64_t{position_} + T > options_.context) {
    Assign(error_msg, "laguna executor: context of " +
                          std::to_string(options_.context) + " exhausted");
    return false;
  }
  if (rows != LogitsRows::kNone && logits == nullptr) {
    Assign(error_msg, "laguna executor: logits output missing");
    return false;
  }
  const TensorRef& embd = model_->weights().token_embd;
  for (std::int32_t token : tokens) {
    if (token < 0 || static_cast<std::uint64_t>(token) >= embd.rows) {
      Assign(error_msg, "laguna executor: token id outside the vocabulary");
      return false;
    }
  }
  host_embed_.resize(T * H);
  for (std::size_t t = 0; t < T; ++t) {
    reference::DequantizeRow(
        embd.type,
        static_cast<const std::uint8_t*>(embd.data) +
            static_cast<std::size_t>(tokens[t]) * embd.RowBytes(),
        host_embed_.data() + t * H, H);
  }
  // From here on a failure can leave some layers' caches advanced.
  if (const hipError_t s =
          hipMemcpy(h_.ptr, host_embed_.data(), T * H * sizeof(float),
                    hipMemcpyHostToDevice);
      s != hipSuccess) {
    return Fail(error_msg, std::string("laguna executor: embedding upload: ") +
                               hipGetErrorString(s));
  }
  if (!Run(static_cast<std::uint32_t>(T), position_, rows, error_msg)) {
    poisoned_ = true;
    return false;
  }
  if (rows != LogitsRows::kNone) {
    const std::size_t count =
        (rows == LogitsRows::kAll ? T : 1) * config.vocab_size;
    if (const hipError_t s = hipMemcpy(
            logits, logits_.ptr, count * sizeof(float), hipMemcpyDeviceToHost);
        s != hipSuccess) {
      return Fail(error_msg, std::string("laguna executor: logits copy: ") +
                                 hipGetErrorString(s));
    }
  }
  position_ += static_cast<std::uint32_t>(T);
  return true;
}

bool Executor::BatchCompatible(const Executor& lead) const noexcept {
  const auto vector_ok = [](const Executor& e) {
    return e.options_.path == Path::kAuto || e.options_.path == Path::kVector;
  };
  return model_ == lead.model_ && vector_ok(*this) && vector_ok(lead);
}

/// Why this executor cannot take `token` now ("" when it can). Nothing is
/// touched by asking.
std::string Executor::RejectDecode(std::int32_t token,
                                   const float* logits) const {
  if (poisoned_) {
    return "laguna executor: a previous call failed; Reset() is required";
  }
  if (logits == nullptr) {
    return "laguna executor: logits output missing";
  }
  const TensorRef& embd = model_->weights().token_embd;
  if (token < 0 || static_cast<std::uint64_t>(token) >= embd.rows) {
    return "laguna executor: token id outside the vocabulary";
  }
  if (std::uint64_t{position_} + 1 > options_.context) {
    return "laguna executor: context of " + std::to_string(options_.context) +
           " exhausted";
  }
  for (const KvCache& cache : kv_) {
    if (!cache.valid() || cache.committed() != position_) {
      return "laguna executor: a KV cache is invalid or out of step; Reset() "
             "is required";
    }
  }
  return {};
}

void Executor::DecodeBatch(std::span<DecodeItem> items) {
  std::vector<DecodeItem*> pending;
  pending.reserve(items.size());
  for (DecodeItem& item : items) {
    item.completed = false;
    item.error.clear();
    if (item.executor == nullptr) {
      item.error = "laguna executor: no executor";
      continue;
    }
    const bool duplicate = std::any_of(
        pending.begin(), pending.end(),
        [&](const DecodeItem* o) { return o->executor == item.executor; });
    if (duplicate) {
      item.error = "laguna executor: executor appears twice in a decode batch";
      continue;
    }
    item.error = item.executor->RejectDecode(item.token, item.logits);
    if (item.error.empty()) {
      pending.push_back(&item);
    }
  }
  // Executors that cannot share a pass with the group's first run in another
  // group (or alone).
  while (!pending.empty()) {
    Executor* lead = pending.front()->executor;
    const std::size_t limit =
        std::min<std::size_t>(kVectorMaxRows, lead->options_.max_chunk);
    std::vector<DecodeItem*> group;
    std::vector<DecodeItem*> rest;
    for (DecodeItem* item : pending) {
      if (group.size() < limit &&
          (item == pending.front() || item->executor->BatchCompatible(*lead))) {
        group.push_back(item);
      } else {
        rest.push_back(item);
      }
    }
    pending = std::move(rest);
    if (group.size() == 1) {
      DecodeItem* item = group.front();
      const std::int32_t token = item->token;
      item->completed = item->executor->Forward(
          std::span<const std::int32_t>(&token, 1), LogitsRows::kLast,
          item->logits, &item->error);
    } else {
      RunGroup(group);
    }
  }
}

void Executor::RunGroup(std::span<DecodeItem* const> group) {
  Executor& lead = *group.front()->executor;
  const Config& config = lead.model_->config();
  const std::size_t H = config.hidden_size;
  const std::size_t n = group.size();
  const std::size_t layers = config.num_layers;
  const auto fail_all = [&](const std::string& message, bool poison) {
    for (DecodeItem* item : group) {
      item->completed = false;
      item->error = message;
      if (poison) {
        item->executor->poisoned_ = true;
      }
    }
  };
  // The only allocation, before anything is enqueued: a failure touches
  // nothing.
  if (lead.rows_.bytes < layers * n * sizeof(DecodeRow)) {
    const std::size_t rows_max =
        std::min<std::size_t>(kVectorMaxRows, lead.options_.max_chunk);
    Buffer fresh;
    std::string alloc_error;
    if (!lead.Alloc(&fresh, layers * rows_max * sizeof(DecodeRow),
                    &alloc_error)) {
      fail_all(alloc_error, false);
      return;
    }
    (void)hipStreamSynchronize(lead.stream_);
    lead.Free(&lead.rows_);
    lead.rows_ = fresh;
  }
  std::vector<Executor*> members(n);
  std::vector<DecodeRow> table(layers * n);
  lead.host_embed_.resize(n * H);
  const TensorRef& embd = lead.model_->weights().token_embd;
  for (std::size_t t = 0; t < n; ++t) {
    Executor& member = *group[t]->executor;
    members[t] = &member;
    for (std::size_t il = 0; il < layers; ++il) {
      table[il * n + t] = AttnOps::DescribeDecodeRow(member.kv_[il]);
    }
    reference::DequantizeRow(
        embd.type,
        static_cast<const std::uint8_t*>(embd.data) +
            static_cast<std::size_t>(group[t]->token) * embd.RowBytes(),
        lead.host_embed_.data() + t * H, H);
  }
  // From here on a failure can leave some layers' caches advanced.
  if (const hipError_t s =
          hipMemcpy(lead.rows_.ptr, table.data(),
                    table.size() * sizeof(DecodeRow), hipMemcpyHostToDevice);
      s != hipSuccess) {
    fail_all(std::string("laguna executor: decode row upload: ") +
                 hipGetErrorString(s),
             true);
    return;
  }
  if (const hipError_t s =
          hipMemcpy(lead.h_.ptr, lead.host_embed_.data(), n * H * sizeof(float),
                    hipMemcpyHostToDevice);
      s != hipSuccess) {
    fail_all(std::string("laguna executor: embedding upload: ") +
                 hipGetErrorString(s),
             true);
    return;
  }
  std::string error;
  if (!lead.Run(static_cast<std::uint32_t>(n), 0, LogitsRows::kAll, &error,
                members)) {
    fail_all(error.empty() ? "laguna batched forward pass failed" : error,
             true);
    return;
  }
  const std::size_t vocab = config.vocab_size;
  for (std::size_t t = 0; t < n; ++t) {
    if (const hipError_t s =
            hipMemcpy(group[t]->logits,
                      static_cast<const float*>(lead.logits_.ptr) + t * vocab,
                      vocab * sizeof(float), hipMemcpyDeviceToHost);
        s != hipSuccess) {
      fail_all(
          std::string("laguna executor: logits copy: ") + hipGetErrorString(s),
          true);
      return;
    }
  }
  for (DecodeItem* item : group) {
    item->executor->position_ += 1;
    item->completed = true;
  }
}

}  // namespace gufo::models::laguna::rocm
