#!/bin/sh
cd build/bin
export OMP_NUM_THREADS=1
srun -N 1 -n 1 -c 16 ./sample_spmv 0 -1 _${1}_1
export OMP_NUM_THREADS=2
srun -N 1 -n 1 -c 16 ./sample_spmv 0 -1 _${1}_2
export OMP_NUM_THREADS=4
srun -N 1 -n 1 -c 16 ./sample_spmv 0 -1 _${1}_4
export OMP_NUM_THREADS=8
srun -N 1 -n 1 -c 16 ./sample_spmv 0 -1 _${1}_8
export OMP_NUM_THREADS=16
srun -N 1 -n 1 -c 16 ./sample_spmv 0 -1 _${1}_16