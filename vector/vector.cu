#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}


void vectorAdd(float *A_h, float *B_h, float *C_h, int n)
{
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    dim3 dimGrid(ceil(n / 256.0), 1, 1); // Same as dimGrid(32) or dimGrid = {32,1,1};
    dim3 dimBlock(256, 1, 1);            // Same as dimBlock(128) or dimBlock = {128,1,1};

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vectorAddKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(void)
{
    int n = 10000000;

    float *A = (float *)malloc(sizeof(float) * n);
    float *B = (float *)malloc(sizeof(float) * n);
    float *O = (float *)malloc(sizeof(float) * n);

    for (int i = 0; i < n; i++)
    {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Benchmark ONLY the vectorAdd function
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    vectorAdd(A, B, O, n);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;

    printf("Vector addition completed in: %f ms\n", elapsed_ms);

    // Print ONLY the first 5 elements for verification
    printf("Verification: O[0] = %f, O[1] = %f, O[2] = %f\n", O[0], O[1], O[2]);

    free(A);
    free(B);
    free(O);

    return 0;
}