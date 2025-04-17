#!/bin/bash

# Number of cores you want to use
NUM_CORES=10
# Which CPU cores to use (e.g., 0-3)
CORE_RANGE="0-9"
GPU_ID=2


# Set thread limits
export OMP_NUM_THREADS=$NUM_CORES
export MKL_NUM_THREADS=$NUM_CORES
export NUMEXPR_NUM_THREADS=$NUM_CORES
export OPENBLAS_NUM_THREADS=$NUM_CORES
export VECLIB_MAXIMUM_THREADS=$NUM_CORES

# Run the training script with taskset

CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c $CORE_RANGE python train.py --config configs/config260k_gpu_1head.yaml &

# CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c $CORE_RANGE python train.py --config configs/config260k_gpu_2head.yaml &

# CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c $CORE_RANGE python train.py --config configs/config260k_gpu_3head.yaml &

# CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c $CORE_RANGE python train.py --config configs/config260k_gpu_4head.yaml &
