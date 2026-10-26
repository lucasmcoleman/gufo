# Lane B owns this fragment; Lane A owns the parent CMakeLists.txt.
target_sources(gufo_glm53_flash_reference PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/reference_mla.cpp
  ${CMAKE_CURRENT_LIST_DIR}/reference_moe.cpp)
set_source_files_properties(reference_mla.cpp reference_moe.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  # Public because executor.hpp exposes the production LaneBOps binding.
  target_compile_definitions(gufo_glm53_flash PUBLIC GLM53_LANE_B_OPS=1)
  set(GLM53_LANE_B_HIP_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/kernels/mmq/iq.hip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/moe.hip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/expert_vec.hip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/mla.hip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/indexer.hip.cpp)
  target_sources(gufo_glm53_flash PRIVATE ${GLM53_LANE_B_HIP_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/lane_b.cpp)
  # Routed IQ2_XXS gate/up reuse DeepSeek V4's vendored llama.cpp MMQ/MMVQ.
  # The mmq archive resolves its Q8 fold hook from the DS4 backend.
  target_link_libraries(gufo_glm53_flash PRIVATE gufo_deepseek_v4_flash_mmq
    gufo_deepseek_v4_flash)
  set_source_files_properties(${GLM53_LANE_B_HIP_SOURCES} PROPERTIES LANGUAGE HIP
    COMPILE_OPTIONS "-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument")
endif()
