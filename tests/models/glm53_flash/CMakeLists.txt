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
