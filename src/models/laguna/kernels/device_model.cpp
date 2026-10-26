#include "src/models/laguna/kernels/device_model.hpp"

#include <hip/hip_runtime.h>

#include <utility>

#include "src/core/hip/weight_upload.hpp"
#include "src/models/laguna/budget.hpp"
#include "src/models/laguna/reference.hpp"

namespace gufo::models::laguna::rocm {
namespace {

using core::GgmlType;

constexpr std::size_t kRowPadding = 512;  // ggml MATRIX_ROW_PADDING

struct Uploader {
  hip::WeightUpload* stager;
  std::vector<void*>* owned;
  std::size_t* bytes;
  std::string* error;

  bool Fail(const std::string& message) const {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  }

  /// Copies `t` (after checking type and shape) with the zero tail the MMQ
  /// tile loaders may read past the last row.
  bool Tensor(const TensorRef& t, GgmlType type, std::uint64_t cols,
              std::uint64_t rows, std::uint64_t experts, const char* what,
              const void** device) const {
    if (t.empty() || t.data == nullptr) {
      return Fail(std::string("laguna upload: missing ") + what);
    }
    if (t.type != type || t.cols != cols || t.rows != rows ||
        t.experts != experts) {
      return Fail(std::string("laguna upload: ") + what +
                  " has an unexpected type or shape");
    }
    const std::size_t size = t.SizeBytes();
    const std::size_t tail =
        t.type == GgmlType::kF32 ? 0 : t.RowBytes() * kRowPadding / t.cols;
    void* ptr = nullptr;
    if (const hipError_t s = hipMalloc(&ptr, size + tail); s != hipSuccess) {
      return Fail(std::string("laguna upload: hipMalloc for ") + what + " (" +
                  std::to_string(size + tail) +
                  " bytes): " + hipGetErrorString(s));
    }
    owned->push_back(ptr);
    *bytes += size + tail;
    if (stager != nullptr) {
      std::string copy_error;
      if (!stager->Copy(t.shard, t.file_offset, size, ptr, &copy_error)) {
        return Fail(std::string("laguna upload of ") + what + ": " +
                    copy_error);
      }
    } else if (const hipError_t s =
                   hipMemcpy(ptr, t.data, size, hipMemcpyHostToDevice);
               s != hipSuccess) {
      return Fail(std::string("laguna upload of ") + what + ": " +
                  hipGetErrorString(s));
    }
    if (tail != 0) {
      if (const hipError_t s =
              hipMemset(static_cast<char*>(ptr) + size, 0, tail);
          s != hipSuccess) {
        return Fail(std::string("laguna upload padding: ") +
                    hipGetErrorString(s));
      }
    }
    *device = ptr;
    return true;
  }

  bool Matrix(const TensorRef& t, GgmlType type, std::uint64_t cols,
              std::uint64_t rows, const char* what, DeviceMatrix* m) const {
    if (!Tensor(t, type, cols, rows, 1, what, &m->data)) {
      return false;
    }
    m->type = type;
    m->cols = static_cast<std::uint32_t>(cols);
    m->rows = static_cast<std::uint32_t>(rows);
    m->experts = 1;
    return true;
  }

  bool F32(const TensorRef& t, std::uint64_t cols, std::uint64_t rows,
           const char* what, const float** device) const {
    const void* p = nullptr;
    if (!Tensor(t, GgmlType::kF32, cols, rows, 1, what, &p)) {
      return false;
    }
    *device = static_cast<const float*>(p);
    return true;
  }
};

bool UploadAttention(const Uploader& up, const Config& config,
                     const LayerWeights& layer, std::uint32_t il,
                     AttnWeights* w) {
  const std::uint64_t hidden = config.hidden_size;
  const std::uint64_t heads = config.n_head(il);
  const std::uint64_t kvh = config.num_kv_heads;
  const std::uint64_t hd = config.head_dim;
  if (hd != 128 || kvh == 0 || heads % kvh != 0) {
    return up.Fail(
        "laguna attention: head_dim must be 128 and heads a "
        "multiple of the kv heads");
  }
  const reference::RopeParams rp = reference::RopeFor(config, il);
  if (rp.n_rot == 0 || rp.n_rot % 2 != 0 || rp.n_rot > hd) {
    return up.Fail("laguna attention: bad rotary dimension");
  }
  if (!up.Matrix(layer.attn_q, GgmlType::kQ6_K, hidden, heads * hd, "attn_q",
                 &w->q) ||
      !up.Matrix(layer.attn_k, GgmlType::kQ6_K, hidden, kvh * hd, "attn_k",
                 &w->k) ||
      !up.Matrix(layer.attn_v, GgmlType::kQ6_K, hidden, kvh * hd, "attn_v",
                 &w->v) ||
      !up.Matrix(layer.attn_output, GgmlType::kQ6_K, heads * hd, hidden,
                 "attn_output", &w->o) ||
      !up.F32(layer.attn_gate, hidden, heads, "attn_gate", &w->gate) ||
      !up.F32(layer.attn_q_norm, hd, 1, "attn_q_norm", &w->q_norm) ||
      !up.F32(layer.attn_k_norm, hd, 1, "attn_k_norm", &w->k_norm)) {
    return false;
  }
  w->hidden = config.hidden_size;
  w->heads = config.n_head(il);
  w->kv_heads = config.num_kv_heads;
  w->head_dim = config.head_dim;
  w->window = config.is_full_attention(il) ? 0U : config.sliding_window;
  w->eps = config.rms_eps;
  const reference::RopeSetup setup = reference::MakeRopeSetup(rp);
  w->rope =
      RopeSpec{rp.n_rot,      setup.theta_scale, setup.corr[0], setup.corr[1],
               rp.freq_scale, rp.ext_factor,     setup.mscale};
  return true;
}

}  // namespace

DeviceModel::~DeviceModel() {
  layers_.clear();  // LayerFfn frees its own tensors
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& weights,
                                                 const core::GgufReader* reader,
                                                 std::string* error_msg) {
  const Config& config = weights.config;
  std::unique_ptr<DeviceModel> model(new DeviceModel());
  model->weights_ = &weights;
  std::unique_ptr<hip::WeightUpload> stager;
  if (reader != nullptr) {
    stager = hip::WeightUpload::Create(reader->GetMappedRegions(), error_msg);
    if (stager == nullptr) {
      return nullptr;
    }
  }
  const Uploader up{stager.get(), &model->allocations_, &model->bytes_,
                    error_msg};
  // Queued copies write into allocations the model owns: drain them before
  // returning on any path, so a failure frees idle memory.
  const auto finish = [&]() {
    if (stager == nullptr) {
      return true;
    }
    std::string drain_error;
    const bool ok = stager->Finish(&drain_error);
    if (!ok && error_msg != nullptr && error_msg->empty()) {
      *error_msg = drain_error;
    }
    return ok;
  };
  const std::uint64_t hidden = config.hidden_size;
  bool ok = up.F32(weights.output_norm, hidden, 1, "output_norm",
                   &model->output_norm_) &&
            up.Matrix(weights.output, GgmlType::kQ6_K, hidden,
                      config.vocab_size, "output", &model->output_);
  model->layers_.resize(config.num_layers);
  for (std::uint32_t il = 0; ok && il < config.num_layers; ++il) {
    const LayerWeights& lw = weights.layers[il];
    DeviceLayer& layer = model->layers_[il];
    ok = up.F32(lw.attn_norm, hidden, 1, "attn_norm", &layer.attn_norm) &&
         up.F32(lw.ffn_norm, hidden, 1, "ffn_norm", &layer.ffn_norm) &&
         UploadAttention(up, config, lw, il, &layer.attn);
    ok = finish() && ok;
    if (!ok) {
      break;
    }
    auto ffn = LayerFfn::Upload(config, lw, error_msg, stager.get());
    if (!ffn) {
      ok = false;
      break;
    }
    model->bytes_ += ffn->device_bytes();
    layer.ffn = std::move(*ffn);
  }
  ok = finish() && ok;
  if (!ok) {
    return nullptr;
  }
  return model;
}

}  // namespace gufo::models::laguna::rocm
