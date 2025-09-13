#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

__global__ void vecadd(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void deviceAlloc(void **addr, size_t size) {
  auto err = cudaMalloc(addr, size);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
    exit(-1);
  }
}

int main() {
  // Display GPU information
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err == cudaSuccess) {
    std::cout << "GPU Device Information:" << std::endl;
    std::cout << "Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock
              << std::endl;
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor
              << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << std::endl;
  } else {
    std::cerr << "Failed to get device properties: " << cudaGetErrorString(err)
              << std::endl;
  }

  constexpr int kN = 100;
  float *h_a = new float[kN];
  float *h_b = new float[kN];
  float *h_c = new float[kN];

  // Initialize input arrays
  std::iota(h_a, h_a + kN, 0); // [0, 1, 2, ..., 99]
  std::iota(h_b, h_b + kN, 0); // [0, 1, 2, ..., 99]

  // Expected result: [0+0, 1+1, 2+2, ..., 99+99] = [0, 2, 4, ..., 198]

  float *d_a, *d_b, *d_c;

  deviceAlloc((void **)&d_a, sizeof(float) * kN);
  deviceAlloc((void **)&d_b, sizeof(float) * kN);
  deviceAlloc((void **)&d_c, sizeof(float) * kN);

  // Copy data to device
  cudaMemcpy(d_a, h_a, sizeof(float) * kN, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(float) * kN, cudaMemcpyHostToDevice);

  // FIXED: Use valid thread block configuration
  // CUDA has limits on threads per block (typically 1024 max)
  const int threadsPerBlock = 256; // Valid block size
  const int blocks =
      (kN + threadsPerBlock - 1) / threadsPerBlock; // Calculate needed blocks

  std::cout << "Launching kernel with " << blocks << " blocks of "
            << threadsPerBlock << " threads each" << std::endl;

  // T4 GPU has compute capability 7.5
  // Use explicit kernel launch with error checking
  vecadd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, kN);

  // Check for kernel launch errors
<<<<<<< Updated upstream
  err = cudaGetLastError();
=======
  cudaError_t err = cudaGetLastError();
>>>>>>> Stashed changes
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
    std::cerr << "Error code: " << err << std::endl;
    std::cerr << "This might be a compute capability compatibility issue."
              << std::endl;
    std::cerr << "T4 GPU requires compute capability 7.5 or higher."
              << std::endl;
    exit(-1);
  }

  // Wait for kernel to complete
  cudaDeviceSynchronize();

  // Check for kernel execution errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel execution failed: " << cudaGetErrorString(err)
              << std::endl;
    exit(-1);
  }

  // Copy result back to host
  cudaMemcpy(h_c, d_c, sizeof(float) * kN, cudaMemcpyDeviceToHost);

  // Print results
  std::cout << "Results (first 10 and last 10 elements):" << std::endl;
  std::cout << "First 10: ";
  for (int i = 0; i < 10; i++) {
    std::cout << h_c[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Last 10:  ";
  for (int i = kN - 10; i < kN; i++) {
    std::cout << h_c[i] << " ";
  }
  std::cout << std::endl;

  // Verify a few results
  std::cout << "Verification:" << std::endl;
  std::cout << "h_c[0] = " << h_c[0] << " (expected: " << h_a[0] + h_b[0] << ")"
            << std::endl;
  std::cout << "h_c[50] = " << h_c[50] << " (expected: " << h_a[50] + h_b[50]
            << ")" << std::endl;
  std::cout << "h_c[99] = " << h_c[99] << " (expected: " << h_a[99] + h_b[99]
            << ")" << std::endl;

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;
}
