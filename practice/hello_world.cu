// multi dimensional threads
#include <stdio.h>
__global__ void hello_GPU(void) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    printf("Hello from GPU[%d][%d]!\n", i, j);
}

int main() {
    #define BLOCK_SIZE 16
    #define GRID_SIZE 1
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);  // now the threads are BLOCK_SIZE*BLOCK_SIZE
    dim3 dimGrid(GRID_SIZE, GRID_SIZE);     // 1 * 1 blocks in a grid
    hello_GPU<<<dimGrid, dimBlock>>>();
    cudaDeviceSynchronize();
    printf("Hello from CPU!\n");
    return 0;
}