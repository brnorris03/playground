# CUDA Matrix Multiplication Makefile

# CUDA compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60 -std=c++11

# cuBLAS library
CUBLAS_LIB = -lcublas

# Targets
all: simple_matmul optimized_matmul optimized_matmul_no_cublas cpu_optimized_matmul test_matmul

# Simple matrix multiplication
simple_matmul: simple_matmul.cu
	$(NVCC) $(NVCC_FLAGS) -o simple_matmul simple_matmul.cu

# Optimized matrix multiplication
optimized_matmul: optimized_matmul.cu
	$(NVCC) $(NVCC_FLAGS) -o optimized_matmul optimized_matmul.cu $(CUBLAS_LIB)

# Optimized matrix multiplication (no cuBLAS)
optimized_matmul_no_cublas: optimized_matmul_no_cublas.cu
	$(NVCC) $(NVCC_FLAGS) -o optimized_matmul_no_cublas optimized_matmul_no_cublas.cu

# CPU-optimized matrix multiplication
cpu_optimized_matmul: cpu_optimized_matmul.cpp
	g++ -O3 -march=native -mavx -mavx2 -mfma -std=c++17 -pthread -o cpu_optimized_matmul cpu_optimized_matmul.cpp

# Test matrix multiplication
test_matmul: test_matmul.cu
	$(NVCC) $(NVCC_FLAGS) -o test_matmul test_matmul.cu

# Clean target
clean:
	rm -f simple_matmul optimized_matmul optimized_matmul_no_cublas cpu_optimized_matmul test_matmul

# Run all implementations
run: all
	@echo "Running Simple Matrix Multiplication..."
	./simple_matmul
	@echo ""
	@echo "Running Optimized Matrix Multiplication (with cuBLAS)..."
	./optimized_matmul
	@echo ""
	@echo "Running Optimized Matrix Multiplication (no cuBLAS)..."
	./optimized_matmul_no_cublas
	@echo ""
	@echo "Running CPU-Optimized Matrix Multiplication..."
	./cpu_optimized_matmul

# Test implementations
test: test_matmul
	@echo "Testing Matrix Multiplication Implementations..."
	./test_matmul

.PHONY: all clean run test
