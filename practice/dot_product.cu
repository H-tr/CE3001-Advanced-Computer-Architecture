#include <stdio.h>

#define SIZE 3

__global__
void dot_prod_cu(int* d_c, int* d_a, int* d_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int temp[SIZE];
    temp[i] = d_a[i] * d_b[i];
    __syncthreads();
    if (i == 0) {
        int sum = 0;
        for (int j = 0; j < SIZE; ++j)
            sum += temp[j];
        *d_c = sum;
    }
}

int main(void) {
    int a[SIZE] = {1, 2, 3};
    int b[SIZE] = {4, 5, 6};
    int c;

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a, sizeof(int) * SIZE);
    cudaMalloc((void**) &d_b, sizeof(int) * SIZE);
    cudaMalloc((void**) &d_c, sizeof(int));

    cudaMemcpy(d_a, a, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
    dot_prod_cu<<<1, 3>>>(d_c, d_a, d_b);
    cudaDeviceSynchronize();

    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("The number of dotprod is: %d\n", c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}