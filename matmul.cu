#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
int K) {
    int j = (blockDim.x * blockIdx.x) + threadIdx.x; // col number
    int i = (blockDim.y * blockIdx.y) + threadIdx.y; // row number
    if (i < M && j < K) {
        float val = 0.0f;
        for (int k = 0; k < N; k++) {
            val += A[(i * N) + k] * B[(k * K) + j];
        }
        C[(i * K) + j] = val;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
