#!/bin/bash

mkdir build-riscv
cd build-riscv
cmake -DCMAKE_TOOLCHAIN_FILE=../riscv.cmake -DCMAKE_BUILD_TYPE=Debug ..
#cmake --build .