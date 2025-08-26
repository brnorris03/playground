#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Tile size for shared memory optimization
// OPTIMIZATION: 32x32 tiles provide optimal balance between:
// - Shared memory usage (2KB per block)
// - Thread occupancy (1024 threads per block)
// - Memory coalescing (32 threads per warp)
// - Cache efficiency (good spatial locality)
#define TILE_SIZE 32

// Optimized matrix multiplication kernel using shared memory tiling
// This implementation uses several key optimizations to maximize performance:
// 1. Shared Memory Tiling: Reduces global memory accesses by ~32x
// 2. Memory Coalescing: Ensures optimal memory bandwidth utilization
// 3. Thread Block Optimization: 32x32 blocks for optimal occupancy
// 4. Proper Synchronization: Ensures data consistency across threads
__global__ void optimizedMatMul(const float *A, const float *B, float *C, int M,
                                int N, int K) {
  // Shared memory tiles for matrix A and B
  // Each thread block loads TILE_SIZE x TILE_SIZE tiles into shared memory
  // This dramatically reduces global memory transactions and improves cache
  // locality
  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  // Thread indices within the block
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Global row and column indices for this thread
  // Each thread computes one element of the output matrix C
  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  // Accumulator for the dot product computation
  float sum = 0.0f;

  // Loop over tiles along the K dimension
  // This implements the tiled matrix multiplication algorithm
  // Each iteration processes a TILE_SIZE x TILE_SIZE tile
  for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    // Calculate the starting column index for this tile
    int tileK = tile * TILE_SIZE;

    // Load tile from matrix A into shared memory
    // OPTIMIZATION: Coalesced memory access pattern
    // - Threads in a warp access consecutive memory locations
    // - This maximizes memory bandwidth utilization (up to 32x improvement)
    // - Each thread loads one element, but the entire warp loads 32 consecutive
    // elements
    if (row < M && tileK + tx < K) {
      sA[ty][tx] = A[row * K + tileK + tx];
    } else {
      sA[ty][tx] = 0.0f; // Zero padding for boundary conditions
    }

    // Load tile from matrix B into shared memory
    // OPTIMIZATION: Coalesced memory access pattern
    // - Similar to A, but accessing B matrix with proper indexing
    // - Ensures optimal memory bandwidth for B matrix access
    if (col < N && tileK + ty < K) {
      sB[ty][tx] = B[(tileK + ty) * N + col];
    } else {
      sB[ty][tx] = 0.0f; // Zero padding for boundary conditions
    }

    // CRITICAL: Synchronize all threads in the block
    // This ensures all threads have finished loading data into shared memory
    // before any thread starts computing with the loaded data
    // Without this, race conditions would occur and results would be incorrect
    __syncthreads();

    // Compute partial dot product using shared memory
    // OPTIMIZATION: Shared memory access is ~100x faster than global memory
    // - Each thread computes one element of the output matrix
    // - Data is reused across multiple threads, reducing global memory traffic
    // - This is the core of the tiling optimization
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += sA[ty][k] * sB[k][tx];
    }

    // CRITICAL: Synchronize again before loading the next tile
    // This prevents threads from overwriting shared memory while others are
    // still reading
    __syncthreads();
  }

  // Write the final result to global memory
  // OPTIMIZATION: Coalesced write pattern
  // - Threads in a warp write to consecutive memory locations
  // - Maximizes memory bandwidth for output writes
  // - Similar to the coalesced read pattern used for loading data
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// Host function for optimized matrix multiplication
// This function orchestrates the GPU computation with optimal configuration
void optimizedMatrixMultiply(const std::vector<float> &A,
                             const std::vector<float> &B, std::vector<float> &C,
                             int M, int N, int K) {
  // Allocate device memory for matrices A, B, and C
  // OPTIMIZATION: Single allocation per matrix reduces memory fragmentation
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  // Copy input matrices from host to device memory
  // OPTIMIZATION: Bulk transfer minimizes PCIe overhead
  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  // Define thread block and grid dimensions
  // OPTIMIZATION: 32x32 thread blocks for optimal occupancy
  // - 1024 threads per block (32*32) is the maximum for NVIDIA GPUs since Fermi
  // - This maximizes GPU utilization and hides memory latency
  // - Note: This limit is NVIDIA-specific; other GPU vendors have different
  // limits
  // - Grid size calculated to cover the entire output matrix
  dim3 blockDim(TILE_SIZE, TILE_SIZE); // 32x32 threads per block
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
               (M + TILE_SIZE - 1) / TILE_SIZE);

  // Launch the optimized matrix multiplication kernel
  // The kernel will automatically utilize all optimizations:
  // - Shared memory tiling
  // - Memory coalescing
  // - Proper synchronization
  optimizedMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

  // Copy the result matrix from device back to host memory
  // OPTIMIZATION: Single bulk transfer minimizes PCIe overhead
  cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Clean up device memory allocations
  // OPTIMIZATION: Proper cleanup prevents memory leaks
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// cuBLAS implementation for performance comparison
// cuBLAS is NVIDIA's highly optimized BLAS library for GPU computation
// This serves as a reference for maximum achievable performance
void cublasMatrixMultiply(const std::vector<float> &A,
                          const std::vector<float> &B, std::vector<float> &C,
                          int M, int N, int K) {
  // Create cuBLAS handle for managing library state
  // OPTIMIZATION: cuBLAS handle enables internal optimizations and caching
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Allocate device memory for cuBLAS computation
  // OPTIMIZATION: cuBLAS may use different memory layouts for optimal
  // performance
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  // Copy input matrices to device memory
  // OPTIMIZATION: cuBLAS expects data to be in device memory
  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  // Perform matrix multiplication using cuBLAS SGEMM
  // OPTIMIZATION: cuBLAS implements highly optimized algorithms including:
  // - Advanced tiling strategies
  // - Tensor core utilization (on supported hardware)
  // - Optimized memory access patterns
  // - Multi-level cache optimization
  // Note: cuBLAS expects column-major format, so we transpose the operation
  const float alpha = 1.0f; // Scaling factor for A*B
  const float beta = 0.0f;  // Scaling factor for C (0 = overwrite C)
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
              &beta, d_C, N);

  // Copy result back to host memory
  // OPTIMIZATION: Single bulk transfer minimizes PCIe overhead
  std::vector<float> C_temp(M * N);
  cudaMemcpy(C_temp.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Transpose result from cuBLAS column-major format back to row-major
  // OPTIMIZATION: This is necessary because cuBLAS uses column-major storage
  // while our implementation uses row-major storage
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = C_temp[j * M + i];
    }
  }

  // Clean up device memory and cuBLAS handle
  // OPTIMIZATION: Proper cleanup prevents memory leaks and resource exhaustion
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);
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
  std::vector<float> C_optimized(M * N);
  std::vector<float> C_cublas(M * N);
  std::vector<float> C_cpu(M * N);

  // Fill matrices with random data
  for (int i = 0; i < M * K; i++) {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Perform optimized CUDA matrix multiplication
  auto start = std::chrono::high_resolution_clock::now();
  optimizedMatrixMultiply(A, B, C_optimized, M, N, K);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  auto optimized_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Perform cuBLAS matrix multiplication
  start = std::chrono::high_resolution_clock::now();
  cublasMatrixMultiply(A, B, C_cublas, M, N, K);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  auto cublas_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Perform CPU matrix multiplication for verification
  start = std::chrono::high_resolution_clock::now();
  cpuMatrixMultiply(A, B, C_cpu, M, N, K);
  end = std::chrono::high_resolution_clock::now();

  auto cpu_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Verify results
  bool optimized_correct = verifyResults(C_optimized, C_cpu, M, N);
  bool cublas_correct = verifyResults(C_cublas, C_cpu, M, N);

  std::cout << "Optimized CUDA Matrix Multiplication Results:" << std::endl;
  std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N
            << std::endl;
  std::cout << "Tile size: " << TILE_SIZE << "x" << TILE_SIZE << std::endl;
  std::cout << "Optimized CUDA time: " << optimized_time.count()
            << " microseconds" << std::endl;
  std::cout << "cuBLAS time: " << cublas_time.count() << " microseconds"
            << std::endl;
  std::cout << "CPU time: " << cpu_time.count() << " microseconds" << std::endl;
  std::cout << "Optimized speedup vs CPU: "
            << static_cast<float>(cpu_time.count()) / optimized_time.count()
            << "x" << std::endl;
  std::cout << "cuBLAS speedup vs CPU: "
            << static_cast<float>(cpu_time.count()) / cublas_time.count() << "x"
            << std::endl;
  std::cout << "Optimized results correct: "
            << (optimized_correct ? "Yes" : "No") << std::endl;
  std::cout << "cuBLAS results correct: " << (cublas_correct ? "Yes" : "No")
            << std::endl;

  return 0;
}
