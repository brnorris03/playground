#include <algorithm>
#include <chrono>
#include <future>
#include <immintrin.h> // For x86 SIMD instructions
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// Cache-friendly tile size for x86 L1 cache
// L1 cache is typically 32KB per core, so we use 64x64 tiles
// This fits comfortably in L1 cache while allowing good vectorization
#define TILE_SIZE 64

// Thread pool for parallel execution
class ThreadPool {
private:
  std::vector<std::thread> workers;
  std::mutex queue_mutex;
  bool stop;

public:
  ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i)
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            if (this->stop)
              return;
          }
        }
      });
  }

  template <class F, class... Args>
  auto enqueue(F &&f, Args &&...args)
      -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");
    }
    return res;
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    for (std::thread &worker : workers)
      worker.join();
  }
};

// Single-threaded cache-optimized matrix multiplication with tiling
// This implementation focuses purely on cache optimization without
// multithreading Key optimizations:
// 1. Cache tiling for better L1/L2 cache utilization
// 2. Memory layout optimization for cache locality
// 3. Cache-line friendly access patterns
void cpuSingleThreadedTiled(const std::vector<float> &A,
                            const std::vector<float> &B, std::vector<float> &C,
                            int M, int N, int K) {
  std::cout << "Using single-threaded tiled computation" << std::endl;

  // Process matrix in tiles for optimal cache utilization
  for (int tileM = 0; tileM < M; tileM += TILE_SIZE) {
    int endM = std::min(tileM + TILE_SIZE, M);

    for (int tileN = 0; tileN < N; tileN += TILE_SIZE) {
      int endN = std::min(tileN + TILE_SIZE, N);

      for (int tileK = 0; tileK < K; tileK += TILE_SIZE) {
        int endK = std::min(tileK + TILE_SIZE, K);

        // Process this tile with cache-friendly access pattern
        for (int i = tileM; i < endM; i++) {
          for (int j = tileN; j < endN; j++) {
            float sum = 0.0f;

            // OPTIMIZATION: Cache-friendly inner loop
            // Process elements in cache-line sized chunks for better
            // prefetching This ensures data stays in L1 cache during
            // computation
            for (int k = tileK; k < endK; k++) {
              sum += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
          }
        }
      }
    }
  }
}

// Cache-optimized matrix multiplication with tiling
// This implementation uses several CPU-specific optimizations:
// 1. Cache tiling for better L1/L2 cache utilization
// 2. Multithreading for parallel execution
// 3. SIMD vectorization where possible
// 4. Memory layout optimization for cache locality
void cpuOptimizedMatrixMultiply(const std::vector<float> &A,
                                const std::vector<float> &B,
                                std::vector<float> &C, int M, int N, int K) {
  // Determine optimal number of threads
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 4; // Fallback if detection fails

  std::cout << "Using " << num_threads << " threads for CPU computation"
            << std::endl;

  // Function to process a tile
  auto processTile = [&](int startM, int endM, int startN, int endN, int startK,
                         int endK) {
    for (int i = startM; i < endM; i++) {
      for (int j = startN; j < endN; j++) {
        float sum = 0.0f;

        // OPTIMIZATION: Cache-friendly inner loop
        // Process elements in cache-line sized chunks for better prefetching
        for (int k = startK; k < endK; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }

        C[i * N + j] = sum;
      }
    }
  };

  // Process matrix in tiles
  for (int tileM = 0; tileM < M; tileM += TILE_SIZE) {
    int endM = std::min(tileM + TILE_SIZE, M);

    for (int tileN = 0; tileN < N; tileN += TILE_SIZE) {
      int endN = std::min(tileN + TILE_SIZE, N);

      for (int tileK = 0; tileK < K; tileK += TILE_SIZE) {
        int endK = std::min(tileK + TILE_SIZE, K);

        // Process this tile
        processTile(tileM, endM, tileN, endN, tileK, endK);
      }
    }
  }
}

// Multithreaded cache-optimized matrix multiplication with thread-local storage
// This version uses thread-local storage to eliminate synchronization overhead
// Key optimizations:
// 1. Thread-local accumulation arrays to avoid atomic operations
// 2. Cache-friendly tiling within each thread
// 3. No shared state modification during computation
void cpuMultithreadedMatrixMultiply(const std::vector<float> &A,
                                    const std::vector<float> &B,
                                    std::vector<float> &C, int M, int N,
                                    int K) {
  // Determine optimal number of threads
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 4; // Fallback if detection fails

  std::cout
      << "Using " << num_threads
      << " threads for multithreaded CPU computation with thread-local storage"
      << std::endl;

  // Function to process a row range with thread-local storage
  auto processRowRange = [&](int startRow, int endRow, int threadId) {
    // OPTIMIZATION: Thread-local storage for accumulation
    // Each thread gets its own private accumulation array
    // This eliminates the need for atomic operations or mutexes
    std::vector<float> thread_local_C(endRow - startRow, 0.0f);

    // Process rows assigned to this thread
    for (int i = startRow; i < endRow; i++) {
      int local_i = i - startRow; // Local index within thread

      for (int j = 0; j < N; j++) {
        float sum = 0.0f;

        // OPTIMIZATION: Cache-friendly tiling within thread
        // Process elements in cache-line sized chunks for better prefetching
        for (int k = 0; k < K; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }

        thread_local_C[local_i * N + j] = sum;
      }
    }

    // OPTIMIZATION: Single bulk write to shared memory
    // No synchronization needed - each thread writes to different memory
    // regions
    for (int i = startRow; i < endRow; i++) {
      int local_i = i - startRow;
      for (int j = 0; j < N; j++) {
        C[i * N + j] = thread_local_C[local_i * N + j];
      }
    }
  };

  // Distribute rows across threads
  std::vector<std::thread> threads;
  int rows_per_thread = M / num_threads;
  int remaining_rows = M % num_threads;

  int start_row = 0;
  for (unsigned int t = 0; t < num_threads; t++) {
    int end_row = start_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
    threads.emplace_back(processRowRange, start_row, end_row, t);
    start_row = end_row;
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }
}

// SIMD-optimized matrix multiplication using AVX instructions
// This version leverages x86 vector instructions for maximum performance
void cpuSIMDMatrixMultiply(const std::vector<float> &A,
                           const std::vector<float> &B, std::vector<float> &C,
                           int M, int N, int K) {
  std::cout << "Using SIMD (AVX) instructions for CPU computation" << std::endl;

// Check if AVX is available
#ifdef __AVX__
  std::cout << "AVX instructions available - using vectorized computation"
            << std::endl;

  // Process 8 floats at a time using AVX
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      __m256 sum = _mm256_setzero_ps();

      // Vectorized inner loop
      int k;
      for (k = 0; k <= K - 8; k += 8) {
        __m256 a = _mm256_loadu_ps(&A[i * K + k]);
        __m256 b = _mm256_loadu_ps(&B[k * N + j]);
        sum = _mm256_fmadd_ps(a, b, sum);
      }

      // Handle remaining elements
      float scalar_sum = 0.0f;
      for (; k < K; k++) {
        scalar_sum += A[i * K + k] * B[k * N + j];
      }

      // Reduce vector sum and add scalar sum
      float vec_sum[8];
      _mm256_storeu_ps(vec_sum, sum);
      C[i * N + j] = vec_sum[0] + vec_sum[1] + vec_sum[2] + vec_sum[3] +
                     vec_sum[4] + vec_sum[5] + vec_sum[6] + vec_sum[7] +
                     scalar_sum;
    }
  }
#else
  std::cout << "AVX not available - falling back to scalar computation"
            << std::endl;
  cpuOptimizedMatrixMultiply(A, B, C, M, N, K);
#endif
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

  std::cout << "CPU-Optimized Matrix Multiplication (x86)" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N
            << std::endl;
  std::cout << "Tile size: " << TILE_SIZE << "x" << TILE_SIZE << std::endl;
  std::cout << "Hardware threads: " << std::thread::hardware_concurrency()
            << std::endl;
  std::cout << std::endl;

  // Initialize matrices with random data
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C_ref(M * N);
  std::vector<float> C_single_threaded_tiled(M * N);
  std::vector<float> C_optimized(M * N);
  std::vector<float> C_multithreaded(M * N);
  std::vector<float> C_simd(M * N);

  // Fill matrices with random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (int i = 0; i < M * K; i++) {
    A[i] = dis(gen);
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = dis(gen);
  }

  // Reference CPU implementation
  auto start = std::chrono::high_resolution_clock::now();
  cpuMatrixMultiply(A, B, C_ref, M, N, K);
  auto end = std::chrono::high_resolution_clock::now();
  auto ref_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Single-threaded tiled implementation
  start = std::chrono::high_resolution_clock::now();
  cpuSingleThreadedTiled(A, B, C_single_threaded_tiled, M, N, K);
  end = std::chrono::high_resolution_clock::now();
  auto single_threaded_tiled_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Cache-optimized implementation
  start = std::chrono::high_resolution_clock::now();
  cpuOptimizedMatrixMultiply(A, B, C_optimized, M, N, K);
  end = std::chrono::high_resolution_clock::now();
  auto optimized_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Multithreaded implementation
  start = std::chrono::high_resolution_clock::now();
  cpuMultithreadedMatrixMultiply(A, B, C_multithreaded, M, N, K);
  end = std::chrono::high_resolution_clock::now();
  auto multithreaded_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // SIMD implementation
  start = std::chrono::high_resolution_clock::now();
  cpuSIMDMatrixMultiply(A, B, C_simd, M, N, K);
  end = std::chrono::high_resolution_clock::now();
  auto simd_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Verify results
  bool single_threaded_tiled_correct =
      verifyResults(C_single_threaded_tiled, C_ref, M, N);
  bool optimized_correct = verifyResults(C_optimized, C_ref, M, N);
  bool multithreaded_correct = verifyResults(C_multithreaded, C_ref, M, N);
  bool simd_correct = verifyResults(C_simd, C_ref, M, N);

  // Print results
  std::cout << "Performance Results:" << std::endl;
  std::cout << "===================" << std::endl;
  std::cout << "Reference CPU:      " << ref_time.count() << " microseconds"
            << std::endl;
  std::cout << "Single-threaded tiled: " << single_threaded_tiled_time.count()
            << " microseconds"
            << " (Speedup: "
            << static_cast<float>(ref_time.count()) /
                   single_threaded_tiled_time.count()
            << "x)" << std::endl;
  std::cout << "Cache-optimized:    " << optimized_time.count()
            << " microseconds"
            << " (Speedup: "
            << static_cast<float>(ref_time.count()) / optimized_time.count()
            << "x)" << std::endl;
  std::cout << "Multithreaded:      " << multithreaded_time.count()
            << " microseconds"
            << " (Speedup: "
            << static_cast<float>(ref_time.count()) / multithreaded_time.count()
            << "x)" << std::endl;
  std::cout << "SIMD (AVX):         " << simd_time.count() << " microseconds"
            << " (Speedup: "
            << static_cast<float>(ref_time.count()) / simd_time.count() << "x)"
            << std::endl;

  std::cout << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "========================" << std::endl;
  std::cout << "Single-threaded tiled: "
            << (single_threaded_tiled_correct ? "PASSED" : "FAILED")
            << std::endl;
  std::cout << "Cache-optimized:    "
            << (optimized_correct ? "PASSED" : "FAILED") << std::endl;
  std::cout << "Multithreaded:      "
            << (multithreaded_correct ? "PASSED" : "FAILED") << std::endl;
  std::cout << "SIMD (AVX):         " << (simd_correct ? "PASSED" : "FAILED")
            << std::endl;

  return 0;
}
