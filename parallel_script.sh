#!/bin/bash
cd build-riscv/bin
export OMP_NUM_THREADS=1
./sample_spmv 0 -1 _riscLMUL8_1
export OMP_NUM_THREADS=2
./sample_spmv 0 -1 _riscLMUL8_2
export OMP_NUM_THREADS=4
./sample_spmv 0 -1 _riscLMUL8_4
export OMP_NUM_THREADS=8
./sample_spmv 0 -1 _riscLMUL8_8