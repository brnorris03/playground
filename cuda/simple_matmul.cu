#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Simple matrix multiplication kernel
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

// Host function to perform matrix multiplication
void simpleMatrixMultiply(const std::vector<float> &A,
                          const std::vector<float> &B, std::vector<float> &C,
                          int M, int N, int K) {
  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  // Define block and grid dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (M + blockDim.y - 1) / blockDim.y);

  // Launch kernel
  simpleMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

  // Copy result back to host
  cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// Reference CPU implementation for verification
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

// Function to verify results
bool verifyResults(const std::vector<float> &C1, const std::vector<float> &C2,
                   int M, int N, float tolerance = 1e-5f) {
  for (int i = 0; i < M * N; i++) {
    if (std::abs(C1[i] - C2[i]) > tolerance) {
      std::cout << "Mismatch at index " << i << ": " << C1[i] << " vs " << C2[i]
                << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  // Matrix dimensions
  int M = 1024, N = 1024, K = 1024;

  // Initialize matrices
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C_simple(M * N);
  std::vector<float> C_cpu(M * N);

  // Fill matrices with random data
  for (int i = 0; i < M * K; i++) {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Perform simple CUDA matrix multiplication
  auto start = std::chrono::high_resolution_clock::now();
  simpleMatrixMultiply(A, B, C_simple, M, N, K);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  auto cuda_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Perform CPU matrix multiplication for verification
  start = std::chrono::high_resolution_clock::now();
  cpuMatrixMultiply(A, B, C_cpu, M, N, K);
  end = std::chrono::high_resolution_clock::now();

  auto cpu_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Verify results
  bool correct = verifyResults(C_simple, C_cpu, M, N);

  std::cout << "Simple CUDA Matrix Multiplication Results:" << std::endl;
  std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N
            << std::endl;
  std::cout << "CUDA time: " << cuda_time.count() << " microseconds"
            << std::endl;
  std::cout << "CPU time: " << cpu_time.count() << " microseconds" << std::endl;
  std::cout << "Speedup: "
            << static_cast<float>(cpu_time.count()) / cuda_time.count() << "x"
            << std::endl;
  std::cout << "Results correct: " << (correct ? "Yes" : "No") << std::endl;

  return 0;
}
