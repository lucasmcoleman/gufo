# Laguna-S-2.1 model package. Phase 1 is CPU only: configuration, weight index
# and tokenizer. The BPE engine comes from the Qwen tokenizer in gufo_core.
add_library(gufo_laguna STATIC
  config.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_laguna PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_laguna PUBLIC gufo_core)
