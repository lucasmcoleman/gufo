# Lane B owns this fragment; Lane A owns the parent CMakeLists.txt.
target_sources(gufo_glm53_flash_reference PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/reference_mla.cpp
  ${CMAKE_CURRENT_LIST_DIR}/reference_moe.cpp)
set_source_files_properties(reference_mla.cpp reference_moe.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")
