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

# Chat template: the compiled renderer against Poolside's Jinja output. No model
# needed. The token check reads the real GGUF and is skipped without it.
add_laguna_test(chat_template gufo_laguna "cpu")
# The tool-call scanner links into the HTTP layer, not gufo_laguna.
target_sources(laguna_chat_template_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/models/laguna/tool_calls.cpp)
target_compile_definitions(laguna_chat_template_test PRIVATE
  GUFO_LAGUNA_CHAT_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_goldens.json")

# Oracle tests run on tiny synthetic models: no model file, nothing skipped.
foreach(name IN ITEMS reference_model reference_attention reference_moe reference_quant)
  add_laguna_test(${name} gufo_laguna "cpu")
endforeach()

# CLI that runs the oracle on the real GGUF (not a test: needs ~90 GiB free).
add_executable(laguna_oracle oracle_cli.cpp)
target_link_libraries(laguna_oracle PRIVATE gufo_laguna)
add_dependencies(laguna_tests laguna_oracle)

# DFlash drafter (phase 6a, CPU only). `dflash_contract` needs no model. `dflash`
# reads the real drafter GGUF header (and the target's, for the pairing) and
# skips (status 77) without them; a skip verifies nothing.
add_laguna_test(dflash_contract gufo_laguna "cpu")
# Oracle for the drafter forward on tiny random models (no model file).
add_laguna_test(dflash_reference gufo_laguna "cpu")

foreach(name IN ITEMS config weights tokenizer_golden dflash)
  add_laguna_test(${name} gufo_laguna "cpu;external-model")
  set_tests_properties(laguna.${name} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
# Opt-in (GUFO_LAGUNA_DFLASH_REAL=1): real drafter on CPU, taps from the
# 74 GB target's CPU oracle. Skips (77) by default and below 90 GiB MemAvailable.
add_laguna_test(dflash_real_smoke gufo_laguna "cpu;external-model;opt-in")
set_tests_properties(laguna.dflash_real_smoke PROPERTIES SKIP_RETURN_CODE 77)
add_laguna_test(chat_template_tokens gufo_laguna "cpu;external-model")
set_tests_properties(laguna.chat_template_tokens PROPERTIES SKIP_RETURN_CODE 77)
target_compile_definitions(laguna_chat_template_tokens_test PRIVATE
  GUFO_LAGUNA_CHAT_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_goldens.json")
target_compile_definitions(laguna_tokenizer_golden_test PRIVATE
  GUFO_LAGUNA_TOKENIZER_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_goldens.json")

# GPU FFN operators (phase 3a) against the CPU oracle. `ffn_ops` runs on random
# tiny layers and skips nothing. `ffn_real` slices the real GGUF, so without it
# the test exits 77 and CTest reports a skip, which verifies nothing.
if (ENGINE_ENABLE_HIP)
  add_laguna_test(ffn_ops gufo_laguna "hip;gpu;gfx1151")
  add_laguna_test(ffn_real gufo_laguna "hip;gpu;gfx1151;external-model")
  set_tests_properties(laguna.ffn_real PROPERTIES SKIP_RETURN_CODE 77)

  # GPU attention operator (phase 3b) against the CPU oracle. `attention_ops`
  # runs on random tiny layers and skips nothing. `attention_real` uses real
  # GGUF layers (full L0 and L4, sliding L1), so without the model it exits 77
  # and CTest reports a skip, which verifies nothing.
  add_laguna_test(attention_ops gufo_laguna "hip;gpu;gfx1151")
  add_laguna_test(attention_real gufo_laguna "hip;gpu;gfx1151;external-model")
  set_tests_properties(laguna.attention_real PROPERTIES SKIP_RETURN_CODE 77)

  # Full-model GPU executor (phase 4) on random tiny models: the oracle match,
  # chunk-split and decode bit-identity, reset, untapped reruns and the
  # allocation-failure paths. Nothing is skipped.
  add_laguna_test(executor gufo_laguna "hip;gpu;gfx1151")

  # Serving engine (phase 5) on tiny random models: determinism, snapshots at
  # the window and ring boundaries, failure-and-reset, concurrent sessions and
  # repeated load/unload. Nothing is skipped.
  add_laguna_test(engine gufo_laguna "hip;gpu;gfx1151")

  # Batched decode (phase 5): several sessions advance in shared forward
  # passes. Bit-identity with solo decode across the window and ring wrap,
  # membership changes, up-front rejections, injected failures, tapped and
  # untapped reruns, two streams. Nothing is skipped.
  add_laguna_test(batch gufo_laguna "hip;gpu;gfx1151")

  # The serve path (InferenceBackend, scheduler, snapshot pool, runner) on a
  # tiny random model: registration, chat/completion against the engine,
  # snapshots, cancellation, several sessions, the context limit and injected
  # device errors. Nothing is skipped.
  add_executable(laguna_serving_test serving_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/component_text_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  target_link_libraries(laguna_serving_test PRIVATE gufo_logging gufo_core
    gufo_deepseek_v4_flash gufo_qwen38_flash_next gufo_laguna gufo_cache_hip
    hip::host)
  add_test(NAME laguna.serving COMMAND laguna_serving_test)
  set_tests_properties(laguna.serving PROPERTIES LABELS "laguna;hip;gpu;gfx1151;server")
  add_dependencies(laguna_tests laguna_serving_test)

  # `gufo bench` rows (pp, tg, depth) and prefill validation on a tiny model.
  add_executable(laguna_bench_test bench_test.cpp)
  target_link_libraries(laguna_bench_test PRIVATE gufo_llm_cli gufo_laguna)
  add_test(NAME laguna.bench COMMAND laguna_bench_test)
  set_tests_properties(laguna.bench PROPERTIES LABELS "laguna;hip;gpu;gfx1151")
  add_dependencies(laguna_tests laguna_bench_test)

  # CLI: teacher-forced all-position logit dump (the #969 contract). Not a
  # test: it needs the 74 GB model resident on the GPU.
  add_executable(laguna_logits logits_cli.cpp)
  target_link_libraries(laguna_logits PRIVATE gufo_laguna)
  add_dependencies(laguna_tests laguna_logits)
endif()

# Functional suites (tests/functional/run.py) for Laguna. `functional_suites`
# checks the wiring and needs no model. `functional` runs them against a server
# on the real model: it skips (status 77) without the GGUF and without
# GUFO_LAGUNA_FUNCTIONAL=1, so a plain `ctest -R '^laguna\.'` never loads
# 68.6 GiB; see tests/functional/laguna_suites.py for the command to run it.
if (Python3_EXECUTABLE)
  add_test(NAME laguna.functional_suites
    COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/tests/functional/test_laguna_suites.py)
  set_tests_properties(laguna.functional_suites PROPERTIES LABELS "laguna;cpu;functional")
  if (TARGET gufo)
    add_test(NAME laguna.functional
      COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/functional/laguna_suites.py --ctest
        --binary $<TARGET_FILE:gufo>
        --output ${CMAKE_BINARY_DIR}/laguna-functional)
    set_tests_properties(laguna.functional PROPERTIES
      LABELS "laguna;gpu;hip;functional;external-model;slow"
      SKIP_RETURN_CODE 77 TIMEOUT 14400)
  endif()
endif()
