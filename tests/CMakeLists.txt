add_executable(test_safetensors test_safetensors.cpp)
target_link_libraries(test_safetensors PRIVATE aie_core)
add_test(NAME test_safetensors COMMAND test_safetensors)

add_executable(test_lut_gemm test_lut_gemm.cpp)
target_link_libraries(test_lut_gemm PRIVATE aie_core)
add_test(NAME test_lut_gemm COMMAND test_lut_gemm)

add_executable(test_paged_kv test_paged_kv.cpp)
target_link_libraries(test_paged_kv PRIVATE aie_core)
add_test(NAME test_paged_kv COMMAND test_paged_kv)
