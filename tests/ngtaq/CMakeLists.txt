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
    benchmark_aqindex
    test_fvecs_io
    benchmark_sift1m
    build_ngt_sift1m
    benchmark_comparison
    benchmark_v2_fast
    benchmark_v2_clean
    benchmark_v2_parallel
    benchmark_single_gamma
    build_ngtaqv2
    rebuild_graph
    refine_graph
    test_vector_record
    test_srht
    test_kmeans
    test_pca
    test_adc_distance
    patch_tombstones
    test_l2_fp16
    test_gpq4_kernel
    test_gpq4_store
)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SRC}.cpp")
        add_executable(${TEST_SRC} ${TEST_SRC}.cpp)
        target_link_libraries(${TEST_SRC} ngt)
        target_compile_features(${TEST_SRC} PRIVATE cxx_std_17)
        if(${TEST_SRC} STREQUAL "benchmark_comparison")
            target_compile_definitions(${TEST_SRC} PRIVATE NGT_QBG_DISABLED)
        endif()
    endif()
endforeach()

# HDF5-dependent targets (ann_bench, build_ngtaqv2_ann)
find_package(HDF5 QUIET COMPONENTS C)
if(NOT HDF5_FOUND)
    find_library(HDF5_C_LIB NAMES hdf5 hdf5_serial hdf5-serial)
    if(HDF5_C_LIB)
        set(HDF5_C_LIBRARIES ${HDF5_C_LIB})
        set(HDF5_INCLUDE_DIRS "/usr/include" "/usr/include/hdf5/serial")
        set(HDF5_FOUND TRUE)
    endif()
endif()

if(HDF5_FOUND)
    message(STATUS "HDF5 found: ${HDF5_C_LIBRARIES} -- building ann_bench, build_ngtaqv2_ann, qg/qbg tools")
    foreach(HDF5_SRC
        ann_bench
        build_ngtaqv2_ann
        build_ngt_ann
        diag_ids
        diag_batch_dist
        diag_route_rank
        diag_route_recall
        diag_2level
        qg_build
        qg_bench
        qbg_build
        qbg_bench
        test_global_pq
    )
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${HDF5_SRC}.cpp")
            add_executable(${HDF5_SRC} ${HDF5_SRC}.cpp)
            target_include_directories(${HDF5_SRC} PRIVATE ${HDF5_INCLUDE_DIRS})
            target_link_libraries(${HDF5_SRC} ngt ${HDF5_C_LIBRARIES})
            target_compile_features(${HDF5_SRC} PRIVATE cxx_std_17)
            # Diagnostic: -DAQ_PROFILE=ON enables searchV2 per-region timers (off by
            # default; normal builds unaffected). AQ_PROFILE is defined on the ngt lib
            # (where searchV2 lives) so the #ifdef-gated timers compile in.
            if(AQ_PROFILE)
                target_compile_definitions(${HDF5_SRC} PRIVATE AQ_PROFILE)
            endif()
        endif()
    endforeach()
else()
    message(WARNING "HDF5 not found -- skipping ann_bench, build_ngtaqv2_ann, and qg/qbg tools")
endif()
