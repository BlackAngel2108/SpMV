message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

# defines
add_definitions("-Driscv")

# указываем компилятор
set(PRJ_COMPILER_PATH "/home/luba/tools/riscv_gcc/bin")
set(CMAKE_CXX_COMPILER "${PRJ_COMPILER_PATH}/riscv64-unknown-linux-gnu-g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -march=rv64gcv -mabi=lp64d")
set(CMAKE_C_COMPILER "${PRJ_COMPILER_PATH}/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -march=rv64gcv -mabi=lp64d")

# Сборка для risc-v qemu должна быть статик
set(CMAKE_EXE_LINKER_FLAGS "-static")
# Указываем архитектуру (опционально)
set(CMAKE_SYSTEM_NAME "Linux")
# set(CMAKE_SYSTEM_PROCESSOR "x86_64")
