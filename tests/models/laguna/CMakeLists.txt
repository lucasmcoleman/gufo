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

# Oracle tests run on tiny synthetic models: no model file, nothing skipped.
foreach(name IN ITEMS reference_model reference_attention reference_moe reference_quant)
  add_laguna_test(${name} gufo_laguna "cpu")
endforeach()

# CLI that runs the oracle on the real GGUF (not a test: needs ~90 GiB free).
add_executable(laguna_oracle oracle_cli.cpp)
target_link_libraries(laguna_oracle PRIVATE gufo_laguna)
add_dependencies(laguna_tests laguna_oracle)

foreach(name IN ITEMS config weights tokenizer_golden)
  add_laguna_test(${name} gufo_laguna "cpu;external-model")
  set_tests_properties(laguna.${name} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
target_compile_definitions(laguna_tokenizer_golden_test PRIVATE
  GUFO_LAGUNA_TOKENIZER_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_goldens.json")
