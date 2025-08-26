// Simple matmul kernel for multiplying A x B = C where
// A is MxK, B is KxN, and C is MxN

__global__ void simpleMatmul(const float *A, const float *B, float *C, int M,
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

// Host function
void simpleMatmulHost(const std::vector<float> &A, const std::vector<float> &B,
                      std::vector<float> &C, int M, int N, int K) {
  // Allocate device memory
  float *dA, *dB, *dC;
  auto numBytesA = M * K * sizeof(float), numBytesB = K * N * sizeof(float),
       numBytesC = M * N * sizeof(float);
  cudaMalloc(&dA, numBytesA);
  cudaMalloc(&dB, numBytesB);
  cudaMalloc(&dC, numBytesC);

  // Copy data to device
  cudaMemcpy(dA, A.data(), numBytesA);
  cudaMemcpy(dB, A.data(), numBytesB);

  // Not accumulating into pre-existing C

  // Define block and grid dimensions
  dim3 blockDim(16, 16);
  // ncols, nrows
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (M + blockDim.y - 1) / blockDim.y);

  // Launch kernel
  simpleMatmul<<<gridDim, blockDim>>>(dA, dB, dC, M, N, K);

  // Copy result back to host
  cudaMemcpy(dC, C.data(), numBytesC);

  // free memory
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

int main() {
  int M = 1024, N = 1024, K = 1024;

  std::vector<float> A(M * K), B(K * N), C(M * N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-10, 10);

  std::for_each(A.begin(), A.end(), [&](float &x) { x = distrib(gen); });
  std::for_each(B.begin(), B.end(), [&](float &x) { x = distrib(gen); });

  auto start = std::chrono::high_resolution_clock::now();
  simpleMatmulHost(A, B, C, M, N, K);
  auto end = std::chrono::high_resolution_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}
