#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Simple matrix multiplication kernel (same as simple_matmul.cu)
__global__ void simpleMatMul(const float *A, const float *B, float *C, int M,
                             int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// Tile size for optimized version
#define TILE_SIZE 16

// Optimized matrix multiplication kernel (same as optimized_matmul.cu)
__global__ void optimizedMatMul(const float *A, const float *B, float *C, int M,
                                int N, int K) {
  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  float sum = 0.0f;

  for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    int tileK = tile * TILE_SIZE;

    if (row < M && tileK + tx < K) {
      sA[ty][tx] = A[row * K + tileK + tx];
    } else {
      sA[ty][tx] = 0.0f;
    }

    if (col < N && tileK + ty < K) {
      sB[ty][tx] = B[(tileK + ty) * N + col];
    } else {
      sB[ty][tx] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; k++) {
      sum += sA[ty][k] * sB[k][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// CPU reference implementation
void cpuMatrixMultiply(const std::vector<float> &A, const std::vector<float> &B,
                       std::vector<float> &C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// Test function for simple implementation
void testSimpleMatMul() {
  int M = 64, N = 64, K = 64;

  std::vector<float> A(M * K, 1.0f);
  std::vector<float> B(K * N, 2.0f);
  std::vector<float> C_simple(M * N);
  std::vector<float> C_cpu(M * N);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (M + blockDim.y - 1) / blockDim.y);

  simpleMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

  // Copy result back
  cudaMemcpy(C_simple.data(), d_C, M * N * sizeof(float),
             cudaMemcpyDeviceToHost);

  // CPU reference
  cpuMatrixMultiply(A, B, C_cpu, M, N, K);

  // Verify results
  bool correct = true;
  for (int i = 0; i < M * N; i++) {
    if (std::abs(C_simple[i] - C_cpu[i]) > 1e-5f) {
      std::cout << "Simple implementation error at " << i << ": " << C_simple[i]
                << " vs " << C_cpu[i] << std::endl;
      correct = false;
      break;
    }
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  std::cout << "Simple implementation test: " << (correct ? "PASSED" : "FAILED")
            << std::endl;
}

// Test function for optimized implementation
void testOptimizedMatMul() {
  int M = 64, N = 64, K = 64;

  std::vector<float> A(M * K, 1.0f);
  std::vector<float> B(K * N, 2.0f);
  std::vector<float> C_optimized(M * N);
  std::vector<float> C_cpu(M * N);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
               (M + TILE_SIZE - 1) / TILE_SIZE);

  optimizedMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

  // Copy result back
  cudaMemcpy(C_optimized.data(), d_C, M * N * sizeof(float),
             cudaMemcpyDeviceToHost);

  // CPU reference
  cpuMatrixMultiply(A, B, C_cpu, M, N, K);

  // Verify results
  bool correct = true;
  for (int i = 0; i < M * N; i++) {
    if (std::abs(C_optimized[i] - C_cpu[i]) > 1e-5f) {
      std::cout << "Optimized implementation error at " << i << ": "
                << C_optimized[i] << " vs " << C_cpu[i] << std::endl;
      correct = false;
      break;
    }
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  std::cout << "Optimized implementation test: "
            << (correct ? "PASSED" : "FAILED") << std::endl;
}

int main() {
  std::cout << "Testing CUDA Matrix Multiplication Implementations..."
            << std::endl;
  std::cout << "=================================================="
            << std::endl;

  testSimpleMatMul();
  testOptimizedMatMul();

  std::cout << "\nAll tests completed!" << std::endl;
  return 0;
}
