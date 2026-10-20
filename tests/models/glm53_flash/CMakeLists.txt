add_custom_target(glm53_flash_tests)

function(add_glm53_flash_test name library labels)
  set(target glm53_flash_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME glm53_flash.${name} COMMAND ${target})
  set_tests_properties(glm53_flash.${name} PROPERTIES
    LABELS "glm53_flash;${labels}")
  add_dependencies(glm53_flash_tests ${target})
endfunction()

add_glm53_flash_test(weights gufo_glm53_flash "cpu;model")
set_tests_properties(glm53_flash.weights PROPERTIES SKIP_RETURN_CODE 77)

# Tokenizer on the real GGUF vocabulary (GLM53_FLASH_GGUF); skips without it.
add_glm53_flash_test(tokenizer gufo_glm53_flash "cpu;model")
set_tests_properties(glm53_flash.tokenizer PROPERTIES SKIP_RETURN_CODE 77)

# Chat template renderer and tool-call parser against jinja2 goldens
# (fixtures/make_chat_template_fixture.py). The tokenizer part runs only with
# GLM53_FLASH_GGUF set; the goldens need no model.
set(GLM53_CHAT_TEMPLATE_GOLDENS
  "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_template_goldens.json")
add_glm53_flash_test(chat_template gufo_glm53_flash "cpu")
target_compile_definitions(glm53_flash_chat_template_test PRIVATE
  GLM53_CHAT_TEMPLATE_GOLDENS="${GLM53_CHAT_TEMPLATE_GOLDENS}")

# Generation loop (generator.hpp) on a scripted ForwardSession and a synthetic
# tokenizer: stop tokens, prefix reuse, streaming tool calls, and the load
# refusal without Lane B's operators (also on GLM53_FLASH_GGUF when set).
add_glm53_flash_test(generator gufo_glm53_flash "cpu")

# HF-generated hyper-connection fixture (fixtures/make_hc_fixture.py).
set(GLM53_HC_FIXTURE "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hc_fixture.bin")
add_glm53_flash_test(hc_reference gufo_glm53_flash_reference "cpu")
target_compile_definitions(glm53_flash_hc_reference_test PRIVATE
  GLM53_HC_FIXTURE="${GLM53_HC_FIXTURE}")

if (ENGINE_ENABLE_HIP)
  add_glm53_flash_test(hc_ops gufo_glm53_flash_reference "hip;gpu;gfx1151")
  target_compile_definitions(glm53_flash_hc_ops_test PRIVATE
    GLM53_HC_FIXTURE="${GLM53_HC_FIXTURE}")
endif()

# HF-generated KDA fixture (fixtures/make_kda_fixture.py).
set(GLM53_KDA_FIXTURE "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/kda_fixture.bin")
add_glm53_flash_test(kda_reference gufo_glm53_flash_reference "cpu")
target_compile_definitions(glm53_flash_kda_reference_test PRIVATE
  GLM53_KDA_FIXTURE="${GLM53_KDA_FIXTURE}")

if (ENGINE_ENABLE_HIP)
  add_glm53_flash_test(kda_ops gufo_glm53_flash_reference "hip;gpu;gfx1151")
  # GPU KDA mixer (executor.hpp) vs ReferenceModel on a synthetic layer.
  add_glm53_flash_test(kda_executor gufo_glm53_flash_reference
    "hip;gpu;gfx1151")
endif()

# Full CPU reference forward vs the HF tiny e2e fixture
# (fixtures/make_e2e_fixture.py).
set(GLM53_E2E_FIXTURE "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/e2e_fixture.bin")
add_glm53_flash_test(reference_e2e gufo_glm53_flash_reference "cpu")
target_compile_definitions(glm53_flash_reference_e2e_test PRIVATE
  GLM53_E2E_FIXTURE="${GLM53_E2E_FIXTURE}")

if (ENGINE_ENABLE_HIP)
  # GPU layer loop (executor.hpp Executor) with host MLA / FFN stand-ins vs
  # ReferenceModel on the e2e fixture model (KDA layers at head dim 128).
  add_glm53_flash_test(executor_loop gufo_glm53_flash_reference
    "hip;gpu;gfx1151")
  target_compile_definitions(glm53_flash_executor_loop_test PRIVATE
    GLM53_E2E_FIXTURE="${GLM53_E2E_FIXTURE}")
  # Generation loop (generator.hpp) on that executor through
  # MakeExecutorSession: greedy tokens, prefix reuse and stop tokens vs the
  # ReferenceModel.
  add_glm53_flash_test(generator_gpu gufo_glm53_flash_reference
    "hip;gpu;gfx1151")
  # Prefill WMMA projection (kernels/prefill_gemm.hpp) vs an FP64 CPU product,
  # within the Q8_1 activation-rounding bound.
  add_glm53_flash_test(prefill_gemm gufo_glm53_flash "hip;gpu;gfx1151")
  set_tests_properties(glm53_flash.prefill_gemm PROPERTIES SKIP_RETURN_CODE 77)
  target_compile_definitions(glm53_flash_generator_gpu_test PRIVATE
    GLM53_E2E_FIXTURE="${GLM53_E2E_FIXTURE}")
endif()

# V3 probe (not a test): CPU ReferenceModel on a real GGUF, dumping every tap
# for comparison with llama.cpp (tools/glm53/reference_probe.cpp).
add_executable(glm53_flash_reference_probe
  ${PROJECT_SOURCE_DIR}/tools/glm53/reference_probe.cpp)
target_link_libraries(glm53_flash_reference_probe PRIVATE
  gufo_glm53_flash_reference)
set_target_properties(glm53_flash_reference_probe PROPERTIES EXCLUDE_FROM_ALL ON)

# KDA executor parity probe (not a test): GPU KdaMixerForward vs the CPU
# ReferenceModel on layers 0-2 of a real GGUF
# (tools/glm53/kda_executor_probe.cpp).
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_kda_executor_probe
    ${PROJECT_SOURCE_DIR}/tools/glm53/kda_executor_probe.cpp)
  target_link_libraries(glm53_flash_kda_executor_probe PRIVATE
    gufo_glm53_flash_reference)
  set_target_properties(glm53_flash_kda_executor_probe PROPERTIES
    EXCLUDE_FROM_ALL ON)
endif()

# Layer-loop parity probe (not a test): GPU Executor on layers 0-2 plus the
# embedding of a real GGUF, host FFN stand-in, vs the CPU ReferenceModel
# (tools/glm53/executor_loop_probe.cpp).
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_executor_loop_probe
    ${PROJECT_SOURCE_DIR}/tools/glm53/executor_loop_probe.cpp)
  target_link_libraries(glm53_flash_executor_loop_probe PRIVATE
    gufo_glm53_flash_reference)
  set_target_properties(glm53_flash_executor_loop_probe PROPERTIES
    EXCLUDE_FROM_ALL ON)
endif()

# Full-model parity probe (not a test): the GPU Executor over all 45 layers of
# a real GGUF, logits vs llama.cpp and the CPU ReferenceModel
# (tools/glm53/model_probe.cpp).
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_model_probe
    ${PROJECT_SOURCE_DIR}/tools/glm53/model_probe.cpp)
  target_link_libraries(glm53_flash_model_probe PRIVATE
    gufo_glm53_flash_reference)
  set_target_properties(glm53_flash_model_probe PROPERTIES EXCLUDE_FROM_ALL ON)
endif()

# Prefill projection micro-benchmark (not a test): tools/glm53/proj_bench.cpp.
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_proj_bench
    ${PROJECT_SOURCE_DIR}/tools/glm53/proj_bench.cpp)
  target_link_libraries(glm53_flash_proj_bench PRIVATE gufo_glm53_flash)
  set_target_properties(glm53_flash_proj_bench PROPERTIES EXCLUDE_FROM_ALL ON)
endif()

# MoE prefill gate/up micro-benchmark (not a test): tools/glm53/moe_bench.cpp.
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_moe_bench
    ${PROJECT_SOURCE_DIR}/tools/glm53/moe_bench.cpp)
  target_link_libraries(glm53_flash_moe_bench PRIVATE gufo_glm53_flash)
  set_target_properties(glm53_flash_moe_bench PROPERTIES EXCLUDE_FROM_ALL ON)
endif()

# NextN Q2_K/Q3_K routed-expert kernels vs a host oracle (not a test):
# tools/glm53/kquant_moe_probe.cpp.
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_kquant_moe_probe
    ${PROJECT_SOURCE_DIR}/tools/glm53/kquant_moe_probe.cpp)
  target_link_libraries(glm53_flash_kquant_moe_probe PRIVATE gufo_glm53_flash)
  set_target_properties(glm53_flash_kquant_moe_probe PROPERTIES EXCLUDE_FROM_ALL ON)
endif()

# NextN (MTP) block: draft agreement and CPU parity (not a test):
# tools/glm53/nextn_probe.cpp.
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_nextn_probe
    ${PROJECT_SOURCE_DIR}/tools/glm53/nextn_probe.cpp)
  target_link_libraries(glm53_flash_nextn_probe PRIVATE gufo_glm53_flash_reference)
  set_target_properties(glm53_flash_nextn_probe PROPERTIES EXCLUDE_FROM_ALL ON)
endif()

# Greedy MTP decode benchmark (not a test): tools/glm53/mtp_bench.cpp.
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_mtp_bench
    ${PROJECT_SOURCE_DIR}/tools/glm53/mtp_bench.cpp)
  target_link_libraries(glm53_flash_mtp_bench PRIVATE gufo_glm53_flash_reference)
  set_target_properties(glm53_flash_mtp_bench PROPERTIES EXCLUDE_FROM_ALL ON)
endif()


# Speculative verify/rollback probe (not a test): tools/glm53/rollback_probe.cpp.
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_rollback_probe
    ${PROJECT_SOURCE_DIR}/tools/glm53/rollback_probe.cpp)
  target_link_libraries(glm53_flash_rollback_probe PRIVATE gufo_glm53_flash_reference)
  set_target_properties(glm53_flash_rollback_probe PROPERTIES EXCLUDE_FROM_ALL ON)
endif()

# V6 KL probe (not a test): full logits for a token set (tools/glm53/kl_probe.cpp).
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_kl_probe
    ${PROJECT_SOURCE_DIR}/tools/glm53/kl_probe.cpp)
  target_link_libraries(glm53_flash_kl_probe PRIVATE gufo_glm53_flash_reference)
  set_target_properties(glm53_flash_kl_probe PROPERTIES EXCLUDE_FROM_ALL ON)
endif()

# Tokenizer parity probe (not a test): encodes a hex-line corpus for
# comparison with llama-tokenize.
add_executable(glm53_flash_tokenize_probe
  ${PROJECT_SOURCE_DIR}/tools/glm53/tokenize_probe.cpp)
target_link_libraries(glm53_flash_tokenize_probe PRIVATE gufo_glm53_flash)
set_target_properties(glm53_flash_tokenize_probe PROPERTIES EXCLUDE_FROM_ALL ON)

# Lane B tests (owned by the Lane B fragment).
include(${CMAKE_CURRENT_SOURCE_DIR}/lane_b_tests.cmake OPTIONAL)

if (ENGINE_ENABLE_HIP)
  # GPU MLA + DSA indexer operator (layer_ops.hpp MlaForward) vs the CPU
  # oracle (ReferenceModel::Mla's math) and the HF mla_attention.npz fixture
  # from GLM53_LANE_B_FIXTURE_DIR; synthetic chunk-invariance and long
  # (> top_k tokens) selection cases. The operator sources and their HIP
  # flags are owned by the library's lane_b.cmake fragment.
  add_glm53_flash_test(mla_ops gufo_glm53_flash_reference "hip;gpu;gfx1151")
  target_compile_definitions(glm53_flash_mla_ops_test PRIVATE
    GLM53_ROCM_BUILD=1
    GLM53_LANE_B_FIXTURE_DIR="${GLM53_LANE_B_FIXTURE_DIR}")
  # The same binary with every prefill path exact: bitwise chunking checks.
  add_test(NAME glm53_flash.mla_ops_exact COMMAND glm53_flash_mla_ops_test)
  set_tests_properties(glm53_flash.mla_ops_exact PROPERTIES
    LABELS "glm53_flash;hip;gpu;gfx1151"
    ENVIRONMENT "GUFO_GLM53_EXACT_PREFILL=1")
endif()


# NextN Q2_K/Q3_K MoE prefill benchmark (not a test): tools/glm53/q2k_moe_bench.cpp.
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_q2k_moe_bench
    ${PROJECT_SOURCE_DIR}/tools/glm53/q2k_moe_bench.cpp)
  target_link_libraries(glm53_flash_q2k_moe_bench PRIVATE gufo_glm53_flash)
  set_target_properties(glm53_flash_q2k_moe_bench PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
