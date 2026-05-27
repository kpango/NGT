cmake_minimum_required(VERSION 3.5)

include_directories(
    "${PROJECT_SOURCE_DIR}/lib"
    "${PROJECT_BINARY_DIR}/lib/"
    "${PROJECT_BINARY_DIR}/lib/NGT"
)

set(TEST_LINK_LIBS ngt)

foreach(TEST_SRC
    test_bq_distance
    test_binary_quantizer
    test_soa_graph
    test_alpha_cg
    test_dabs
    test_aqindex
    benchmark_aqindex
)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SRC}.cpp")
        add_executable(${TEST_SRC} ${TEST_SRC}.cpp)
        target_link_libraries(${TEST_SRC} ${TEST_LINK_LIBS})
        target_compile_features(${TEST_SRC} PRIVATE cxx_std_17)
    endif()
endforeach()
