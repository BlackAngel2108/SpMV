cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME SpMV_project)
project(${PROJECT_NAME})

#set(CMAKE_CXX_COMPILER "/common/software/gcc-9.5.0/bin/g++")
#set(CMAKE_CXX_COMPILER "C:/msys64/mingw64/bin/g++")

# Добавляем флаг -mavx
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512dq")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vl")




if(COMPILER_SUPPORTS_CXX17)
    message( STATUS "CXX17_supported")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

if(COMPILER_SUPPORTS_CXX17)
    message( STATUS "CXX17_supported")
endif()
#find_package(OpenMP REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP C++ FOUND - ${OpenMP_CXX_VERSION}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        else()
            message(WARNING "OpenMP NOT found")
endif()
#if(OPENMP_FOUND)
#    message( STATUS "OpenMP_found")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")  #${OpenMP_CXX_FLAGS}")
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

# TODO(Kornyakov): not sure if these lines are needed
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})

set(MP2_LIBRARY "${PROJECT_NAME}")
#set(MP2_CUSTOM "${PROJECT_NAME}")
set(MP2_TESTS   "test_${PROJECT_NAME}")
set(MP2_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories("${MP2_INCLUDE}" gtest)

# BUILD
#add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(samples)
add_subdirectory(matrix_binary_library)
add_subdirectory(gtest)
add_subdirectory(test)

# REPORT
message( STATUS "")
message( STATUS "General configuration for ${PROJECT_NAME}")
message( STATUS "======================================")
message( STATUS "")
message( STATUS "   Configuration: ${CMAKE_BUILD_TYPE}")
message( STATUS "")

if( MSVC ) # VS doesn't support correctly the tuples yet
  add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
endif()
