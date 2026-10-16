# GLM-5.3-Flash (glm5-next) model package: config and weight binding first;
# kernels, executor and the CPU oracle are added per lane.
add_library(gufo_glm53_flash STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_glm53_flash PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_glm53_flash PUBLIC gufo_core)

# Scalar oracles, built only by the tests that need them. -O2 in every
# configuration, as for the qwen38_flash_next reference.
add_library(gufo_glm53_flash_reference STATIC EXCLUDE_FROM_ALL
  reference_hc.cpp
  reference_kda.cpp
  reference.cpp)
target_link_libraries(gufo_glm53_flash_reference PUBLIC gufo_glm53_flash)
set_source_files_properties(reference_hc.cpp reference_kda.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(GLM53_HIP_SOURCES kernels/hc.hip.cpp kernels/kda.hip.cpp)
  target_sources(gufo_glm53_flash PRIVATE ${GLM53_HIP_SOURCES})
  set_source_files_properties(${GLM53_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
  target_compile_definitions(gufo_glm53_flash PRIVATE GLM53_ROCM_BUILD=1)
  target_compile_options(gufo_glm53_flash PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_glm53_flash PUBLIC hip::host)
endif()

# Lane B (MLA, DSA indexer, MoE) sources live in a fragment its owner edits;
# OPTIONAL keeps this package building before the fragment exists.
include(${CMAKE_CURRENT_SOURCE_DIR}/lane_b.cmake OPTIONAL)
