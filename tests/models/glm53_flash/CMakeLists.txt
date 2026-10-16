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
endif()

# Full CPU reference forward vs the HF tiny e2e fixture
# (fixtures/make_e2e_fixture.py).
set(GLM53_E2E_FIXTURE "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/e2e_fixture.bin")
add_glm53_flash_test(reference_e2e gufo_glm53_flash_reference "cpu")
target_compile_definitions(glm53_flash_reference_e2e_test PRIVATE
  GLM53_E2E_FIXTURE="${GLM53_E2E_FIXTURE}")

# Lane B tests (owned by the Lane B fragment).
include(${CMAKE_CURRENT_SOURCE_DIR}/lane_b_tests.cmake OPTIONAL)
