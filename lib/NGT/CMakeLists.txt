if( ${UNIX} )
	option(NGT_SHARED_MEMORY_ALLOCATOR "enable shared memory" OFF)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/defines.h.in ${CMAKE_CURRENT_BINARY_DIR}/defines.h)
	# Build-dir paths come first so the generated defines.h (with correct cmake-set flags)
	# takes precedence over the source-tree defines.h fallback.
	include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/lib/" "${PROJECT_SOURCE_DIR}/lib")

	if(${NGT_QBG_DISABLED})
		file(GLOB NGT_SOURCES *.cpp NGTAQ/*.cpp)
	else()
		file(GLOB NGT_SOURCES *.cpp NGTQ/*.cpp NGTAQ/*.cpp)
	endif()
	file(GLOB HEADER_FILES *.h *.hpp)
	file(GLOB NGTQ_HEADER_FILES NGTQ/*.h NGTQ/*.hpp)
	file(GLOB NGTAQ_HEADER_FILES NGTAQ/*.h NGTAQ/*.hpp)

	add_library(ngtstatic STATIC ${NGT_SOURCES})
	set_target_properties(ngtstatic PROPERTIES OUTPUT_NAME ngt)
	set_target_properties(ngtstatic PROPERTIES COMPILE_FLAGS "-fPIC")
	target_link_libraries(ngtstatic)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
		target_link_libraries(ngtstatic OpenMP::OpenMP_CXX)
	endif()

	add_library(ngt SHARED ${NGT_SOURCES})
	set_target_properties(ngt PROPERTIES VERSION ${ngt_VERSION})
	set_target_properties(ngt PROPERTIES SOVERSION ${ngt_SOVERSION})
	# NGTAQ requires C++17 (shared_mutex, unordered_dense, etc.)
	target_compile_features(ngt PRIVATE cxx_std_17)
	target_compile_features(ngtstatic PRIVATE cxx_std_17)
	add_dependencies(ngt ngtstatic)
        if(NOT DEFINED NGT_QBG_DISABLED OR (NOT ${NGT_QBG_DISABLED}))
		# Use CMake-resolved BLAS/LAPACK variables set by find_package() in the top-level CMakeLists.
		# Avoids -llapack/-lblas linker names which may differ across distros/installations.
		target_link_libraries(ngt ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
	endif()
	if(${APPLE})
		target_link_libraries(ngt OpenMP::OpenMP_CXX)
	else(${APPLE})
		# Use CMake-resolved OpenMP target so clang links libomp, gcc links gomp
		if(OpenMP_CXX_FOUND)
			target_link_libraries(ngt OpenMP::OpenMP_CXX rt)
		else()
			target_link_libraries(ngt rt)
		endif()
	endif(${APPLE})

	add_custom_command(OUTPUT command DEPENDS ${NGT_SOURCES} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND sh ${PROJECT_SOURCE_DIR}/utils/mk_version_defs_h.sh ${PROJECT_SOURCE_DIR} version_defs.h)
	add_custom_target(versiondef DEPENDS command)
	add_dependencies(ngtstatic versiondef)
	install(TARGETS
		ngt
		ngtstatic
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib)
	install(FILES ${HEADER_FILES} ${CMAKE_CURRENT_BINARY_DIR}/defines.h  ${CMAKE_CURRENT_BINARY_DIR}/version_defs.h DESTINATION include/NGT)
	install(FILES ${NGTQ_HEADER_FILES} DESTINATION include/NGT/NGTQ)
	install(FILES ${NGTAQ_HEADER_FILES} DESTINATION include/NGT/NGTAQ)
endif()
