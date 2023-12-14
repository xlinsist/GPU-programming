#!/bin/bash
nvcc -arch=sm_70 --cudart shared -o test test.cu
source /home/course/gpgpu-sim_distribution/setup_environment
./test > test_log.txt

