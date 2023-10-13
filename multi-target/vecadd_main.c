#include <stdlib.h>

void vecadd(void *sramPtr);
int main(int argc, char **argv) {
    void *sramPtr = malloc(3 * 6 * sizeof(float) + 1 * sizeof(float));
    // TODO: read inputs from files
    
    vecadd(sramPtr);
    // TODO: write output to file
}
