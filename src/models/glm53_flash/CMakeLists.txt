# GLM-5.3-Flash (glm5-next) model package: config and weight binding first;
# kernels, executor and the CPU oracle are added per lane.
add_library(gufo_glm53_flash STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_glm53_flash PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_glm53_flash PUBLIC gufo_core)
