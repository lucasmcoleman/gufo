# Lane B owns this fragment; Lane A owns the parent CMakeLists.txt.
target_sources(gufo_glm53_flash_reference PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/reference_mla.cpp
  ${CMAKE_CURRENT_LIST_DIR}/reference_moe.cpp)
set_source_files_properties(reference_mla.cpp reference_moe.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(GLM53_LANE_B_HIP_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/kernels/mmq/iq.hip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/moe.hip.cpp)
  target_sources(gufo_glm53_flash PRIVATE ${GLM53_LANE_B_HIP_SOURCES})
  set_source_files_properties(${GLM53_LANE_B_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
endif()
