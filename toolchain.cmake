message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

# Добавляем флаг оптимизации
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Проверяем, какой компилятор использовать
if(DEFINED VARIABLE_NAME AND VARIABLE_NAME STREQUAL "icx")
    # Указываем компилятор C++
    set(CMAKE_CXX_COMPILER "icx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=5")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report-phase=vec")
else()
    set(CMAKE_CXX_COMPILER "g++")  # Используем g++ вместо gcc для C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-optimized")
endif()

# Указываем архитектуру (опционально)
# set(CMAKE_SYSTEM_NAME "Linux")
# set(CMAKE_SYSTEM_PROCESSOR "x86_64")
