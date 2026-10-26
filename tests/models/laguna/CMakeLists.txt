# Laguna-S-2.1 phase 1 tests (CPU only). `contract` needs no model and runs in
# the hosted suite. The other three read the real GGUF header, so without it
# they exit 77 and CTest reports them as skipped; a skip verifies nothing.
add_custom_target(laguna_tests)

function(add_laguna_test name library labels)
  set(target laguna_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME laguna.${name} COMMAND ${target})
  set_tests_properties(laguna.${name} PROPERTIES LABELS "laguna;${labels}")
  add_dependencies(laguna_tests ${target})
endfunction()

add_laguna_test(contract gufo_laguna "cpu")

foreach(name IN ITEMS config weights tokenizer_golden)
  add_laguna_test(${name} gufo_laguna "cpu;external-model")
  set_tests_properties(laguna.${name} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
target_compile_definitions(laguna_tokenizer_golden_test PRIVATE
  GUFO_LAGUNA_TOKENIZER_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_goldens.json")
