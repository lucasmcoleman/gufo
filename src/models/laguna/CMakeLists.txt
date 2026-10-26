# Laguna-S-2.1 model package. CPU only so far: configuration, weight index,
# tokenizer and the F32/Q8-aware reference forward (the oracle).
# The BPE engine comes from the Qwen tokenizer in gufo_core.
add_library(gufo_laguna STATIC
  config.cpp
  reference.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_laguna PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_laguna PUBLIC gufo_core)
