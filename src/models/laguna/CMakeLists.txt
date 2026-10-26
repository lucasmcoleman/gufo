# Laguna-S-2.1 model package: configuration, weight index, tokenizer, the
# F32/Q8-aware reference forward (the oracle) and, with HIP, the GPU FFN
# operators (phase 3a).
# The BPE engine comes from the Qwen tokenizer in gufo_core.
add_library(gufo_laguna STATIC
  config.cpp
  reference.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_laguna PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_laguna PUBLIC gufo_core)

if (ENGINE_ENABLE_HIP)
  # The routed and dense projections reuse the quantized GEMM kernels vendored
  # with qwen38_flash_next. The router and combine kernels are plain IEEE
  # (no fast-math) so their rounding matches the CPU oracle.
  set(LAGUNA_FFN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ffn.hip.cpp)
  target_sources(gufo_laguna PRIVATE ${LAGUNA_FFN_SOURCE})
  set_source_files_properties(${LAGUNA_FFN_SOURCE} PROPERTIES LANGUAGE HIP)
  target_compile_options(gufo_laguna PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_laguna PUBLIC
    gufo_qwen38_flash_next_mmq hip::host)
endif()
