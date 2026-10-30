# CUDA-project
A CUDA-based benchmark project that compares CPU and GPU performance using parallel computing techniques.

# CUDA-Based Parallel Computing Benchmark

## About the Project

This project focuses on comparing CPU and GPU performance using parallel computing techniques with NVIDIA CUDA. It explores how computational workloads can be executed on CPUs and GPUs and evaluates their execution times.

## Objectives

- Understand parallel computing using CUDA.
- Implement computational tasks using CPU and GPU.
- Compare CPU execution time with GPU execution time.
- Analyze the impact of data transfer overhead on GPU performance.
- Study performance differences across computational workloads.

## Technologies Used

- C++
- NVIDIA CUDA
- Google Colab
- NVIDIA GPU (Tesla T4)

## Current Implementation

- GPU detection and CUDA environment verification.
- Basic CUDA vector addition.
- CPU-based vector addition.
- CPU vs GPU execution-time comparison.
- Verification of computational results.

## Preliminary Results

The initial vector addition benchmark with 1,000,000 elements showed that CPU execution was faster than GPU total execution time.

This highlights the impact of memory transfer and kernel launch overhead on GPU performance for relatively simple operations.

## Future Scope

- Benchmark different input sizes.
- Implement matrix multiplication.
- Explore image processing workloads.
- Measure GPU kernel time separately from total execution time.
- Generate performance comparison graphs.
- Analyze GPU speedup and efficiency.

## Execution Environment

The project is developed and tested using Google Colab with an NVIDIA Tesla T4 GPU and CUDA Toolkit.

## Author

Sumati Santosh Dani

---

**Note:** This project is under development. More workloads and performance optimizations will be added progressively.
