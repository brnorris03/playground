// Simple vecadd function
#include <stdint.h>

struct VectorAddPackedArgs {
  float *A;
  float *B;
  float alpha;
  float *C;
};

void vecadd(void *sramPtr) {
  // hard-coded for 6 elements
  int numElements = 6;
  float *inputA = (float*)(sramPtr);
  float *inputB = inputA + numElements;
  float *alpha = inputB + numElements;
  float *outputC = alpha + 1;

  for (int i = 0; i < numElements; i++) {
    outputC[i] = inputA[i] + inputB[i] * *alpha;
  }
}
