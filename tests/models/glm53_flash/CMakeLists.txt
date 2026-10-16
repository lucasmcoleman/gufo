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
# (tools/glm53/kda_executor_probe.cpp, see /server/staging/glm53-gufo/a3).
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
# (tools/glm53/executor_loop_probe.cpp, see /server/staging/glm53-gufo/a3/loop).
if (ENGINE_ENABLE_HIP)
  add_executable(glm53_flash_executor_loop_probe
    ${PROJECT_SOURCE_DIR}/tools/glm53/executor_loop_probe.cpp)
  target_link_libraries(glm53_flash_executor_loop_probe PRIVATE
    gufo_glm53_flash_reference)
  set_target_properties(glm53_flash_executor_loop_probe PROPERTIES
    EXCLUDE_FROM_ALL ON)
endif()

# Tokenizer parity probe (not a test): encodes a hex-line corpus for
# comparison with llama-tokenize (see /server/staging/glm53-gufo/a6).
add_executable(glm53_flash_tokenize_probe
  ${PROJECT_SOURCE_DIR}/tools/glm53/tokenize_probe.cpp)
target_link_libraries(glm53_flash_tokenize_probe PRIVATE gufo_glm53_flash)
set_target_properties(glm53_flash_tokenize_probe PROPERTIES EXCLUDE_FROM_ALL ON)

# Lane B tests (owned by the Lane B fragment).
include(${CMAKE_CURRENT_SOURCE_DIR}/lane_b_tests.cmake OPTIONAL)
