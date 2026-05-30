# Install script for directory: /home/kpango/go/src/github.com/kpango/NGT/lib/NGT

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT/libngt.so.2.5.1"
    "/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT/libngt.so.2"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libngt.so.2.5.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libngt.so.2"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/llvm-strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT/libngt.so")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT/libngt.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/NGT" TYPE FILE FILES
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/ArrayFile.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Capi.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Clustering.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Command.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Common.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Graph.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/GraphOptimizer.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/GraphReconstructor.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/HashBasedBooleanSet.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Index.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/MmapManager.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/MmapManagerDefs.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/MmapManagerException.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/MmapManagerImpl.hpp"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Node.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/ObjectRepository.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/ObjectSpace.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/ObjectSpaceRepository.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Optimizer.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/PrimitiveComparator.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/PrimitiveComparatorNoArch.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/PrimitiveComparatorX86.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/SharedMemoryAllocator.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Thread.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Tree.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/Version.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/defines.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/half.hpp"
    "/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT/defines.h"
    "/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT/version_defs.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/NGT/NGTQ" TYPE FILE FILES
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTQ/Capi.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTQ/HierarchicalKmeans.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTQ/Matrix.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTQ/ObjectFile.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTQ/Optimizer.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTQ/QbgCli.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTQ/QuantizedBlobGraph.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTQ/QuantizedGraph.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTQ/Quantizer.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/NGT/NGTAQ" TYPE FILE FILES
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/ADCDistance.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/ADCTable.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/AQIndex.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/AlphaCGPruner.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/BQDistance.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/BinaryQuantizer.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/DABSSearcher.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/DimUtils.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/KMeansCentering.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/PCAProjector.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/SIMDUtils.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/SRHT.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/SoAGraph.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/VectorRecord.h"
    "/home/kpango/go/src/github.com/kpango/NGT/lib/NGT/NGTAQ/unordered_dense.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/kpango/go/src/github.com/kpango/NGT/build_ngtaq/lib/NGT/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
