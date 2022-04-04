#include <stdio.h>

__global__ 
void vector_add_cu(int* d_c, int* d_a, int* d_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_c[i] = d_a[i] + d_b[i];
}

int main(void) {
    int N = 3;
    int a[N] = {7, 2, 3};
    int b[N] = {6, 4, 5};
    int c[N];

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(int)*N);
    cudaMalloc((void**)&d_b, sizeof(int)*N);
    cudaMalloc((void**)&d_c, sizeof(int)*N);

    cudaMemcpy(d_a, a, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int)*N, cudaMemcpyHostToDevice);

    vector_add_cu<<<1, 3>>>(d_c, d_a, d_b);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, sizeof(int)*N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        printf("%d ", c[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}