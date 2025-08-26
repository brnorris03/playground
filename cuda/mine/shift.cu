__global__ void shiftLeft(float *a, int n) {
  // A is in global memory
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    int temp = a[idx + 1];
    __syncthreads();
    a[idx] = temp;
    __syncthreads();
  }
}

// In addition to the explicit barriers used for thread synchronization within
// a block, there is implicit synchronization at the end of a kernel.

// The host launches the kernel asyncronously, so the kernel can run in parallel
// with the host.

int main(void) {
  float *a = new float[16 * 1024];
  for (int i = 0; i < 16 * 1024; i++) {
    a[i] = i;
  }

  float *d_a;
  cudaMalloc(&d_a, 1024 * sizeof(float));
  dim3 blockDim(1024);
  dim3 gridDim(16);
  shiftLeft<<<gridDim, blockDim>>>(d_a, 16 * 1024);
  cudaDeviceSynchronize();
  cudaFree(d_a);
}
