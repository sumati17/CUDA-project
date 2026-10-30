#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

__global__ void vectorAdd(
    const float* A, const float* B, float* C, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1000000;
    size_t bytes = N * sizeof(float);

    vector<float> A(N, 10.0f);
    vector<float> B(N, 20.0f);
    vector<float> C_cpu(N);
    vector<float> C_gpu(N);

    // CPU benchmark
    auto cpu_start = high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        C_cpu[i] = A[i] + B[i];
    }

    auto cpu_stop = high_resolution_clock::now();

    double cpu_ms =
        duration<double, milli>(cpu_stop - cpu_start).count();

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // GPU benchmark, including memory transfers
    cudaDeviceSynchronize();
    auto gpu_start = high_resolution_clock::now();

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << "Kernel launch error: "
             << cudaGetErrorString(error) << endl;
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    auto gpu_stop = high_resolution_clock::now();

    double gpu_ms =
        duration<double, milli>(gpu_stop - gpu_start).count();

    bool correct = true;

    for (int i = 0; i < N; i++) {
        if (C_cpu[i] != C_gpu[i]) {
            correct = false;
            break;
        }
    }

    cout << "\n--- Vector Addition Benchmark ---\n";
    cout << "Elements: " << N << '\n';
    cout << "CPU time: " << cpu_ms << " ms\n";
    cout << "GPU total time: " << gpu_ms << " ms\n";
    cout << "Results correct: " << (correct ? "Yes" : "No") << '\n';

    if (gpu_ms > 0) {
        cout << "CPU/GPU total-time ratio: "
             << cpu_ms / gpu_ms << "x\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return correct ? 0 : 1;
}
