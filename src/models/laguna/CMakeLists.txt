# Laguna-S-2.1 model package: configuration, weight index, tokenizer, the
# F32/Q8-aware reference forward (the oracle) and, with HIP, the GPU FFN
# operators (phase 3a), the attention operator (phase 3b), the
# full-model executor (phase 4), the serving engine (phase 5) and the DFlash
# drafter's GPU operators (phase 6c), its target side (phase 6d) and the
# speculative decode loop (phase 6e).
# The BPE engine comes from the Qwen tokenizer in gufo_core.
add_library(gufo_laguna STATIC
  budget.cpp
  chat_template.cpp
  config.cpp
  dflash.cpp
  dflash_reference.cpp
  reference.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_laguna PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_laguna PUBLIC gufo_core)

if (ENGINE_ENABLE_HIP)
  # The routed, dense and attention projections reuse the quantized GEMM kernels vendored
  # with qwen38_flash_next. The router and combine kernels are plain IEEE
  # (no fast-math) so their rounding matches the CPU oracle.
  set(LAGUNA_HIP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ffn.hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention.hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/dflash_ops.hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/dflash_session.hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/device_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/executor.hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/speculative.cpp)
  target_sources(gufo_laguna PRIVATE ${LAGUNA_HIP_SOURCES})
  set_source_files_properties(${LAGUNA_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
  target_compile_options(gufo_laguna PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_laguna PUBLIC
    gufo_qwen38_flash_next_mmq hip::host)
endif()
