include_directories(
    "${PROJECT_SOURCE_DIR}/lib"
    "${PROJECT_BINARY_DIR}/lib/"
    "${PROJECT_BINARY_DIR}/lib/NGT"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
link_directories("${PROJECT_BINARY_DIR}/lib/NGT")

foreach(TEST_SRC
    test_bq_distance
    test_binary_quantizer
    test_soa_graph
    test_alpha_cg
    test_dabs
    test_aqindex
    test_fvecs_io
    benchmark_aqindex
    benchmark_sift1m
    build_ngt_sift1m
)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SRC}.cpp")
        add_executable(${TEST_SRC} ${TEST_SRC}.cpp)
        target_link_libraries(${TEST_SRC} ngt)
        target_compile_features(${TEST_SRC} PRIVATE cxx_std_17)
    endif()
endforeach()
