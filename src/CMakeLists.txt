set(target ${MP2_LIBRARY})

file(GLOB hdrs "*.h*" "${MP2_INCLUDE}/*.h*")
file(GLOB srcs "*.cpp")

add_library(${target} ${srcs} ${hdrs})
#target_link_libraries(${target} ${MP2_LIBRARY})


set_property(TARGET ${MP2_LIBRARY} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${MP2_LIBRARY} PROPERTY CXX_STANDARD_REQUIRED ON)
#target_link_libraries(${target} stdc++)

find_package(MKL REQUIRED)

# Simple vars
set(MKL_INCLUDE_DIRS $ENV{MKLROOT}/include)
set(ONEAPI_INCLUDE_DIRS $ENV{ONEAPI_ROOT}/compiler/latest/include)

message("MKL_INCLUDE_DIRS: ${MKL_INCLUDE_DIRS}")
message("ONEAPI_INCLUDE_DIRS: ${ONEAPI_INCLUDE_DIRS}")

# Подключение MKL
target_include_directories(${target} PRIVATE ${MKL_INCLUDE_DIRS})
link_directories($ENV{MKLROOT}/latest/lib)
target_link_libraries(${target} PRIVATE MKL::MKL)

#target_link_directories(${target} PUBLIC /common/software/intel/oneapi_2025.0/mkl/2025.0/bin)
#target_link_libraries(${target} PUBLIC libmkl)
#include_directories(/common/software/intel/oneapi/mkl/2022.1.0/include)
#include_directories(/common/software/intel/oneapi_2025.0/mkl/2025.0/include)
