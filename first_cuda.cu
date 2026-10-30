#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// This function runs on the GPU
__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 10;
    int A[N], B[N], C[N];

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));

    cudaMemcpy(d_A, A, N * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int),
               cudaMemcpyHostToDevice);

    vectorAdd<<<1, 256>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, N * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        cout << A[i] << " + " << B[i]
             << " = " << C[i] << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
