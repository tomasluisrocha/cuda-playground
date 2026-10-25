#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void MatrixVectorMultKernel(float *M, float *V, float *O, size_t width)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < width)
    {
        float rowVal = 0;
        for (int col = 0; col < width; col++)
        {
            rowVal += M[row * width + col] * V[col];
        }
        O[row] = rowVal;
    }
}

void MatrixVectorMult(const std::vector<float> &M_h, const std::vector<float> &V_h, std::vector<float> &O_h, const size_t width)
{
    dim3 dimBlock(32);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x); // Same as ceil(width / dimBlock.x)

    size_t vecSize = width * sizeof(float);
    size_t matSize = width * width * sizeof(float);

    float *M_d, *V_d, *O_d;
    cudaMalloc((void **)&M_d, matSize);
    cudaMalloc((void **)&V_d, vecSize);
    cudaMalloc((void **)&O_d, vecSize);

    cudaMemcpy(M_d, M_h.data(), matSize, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_h.data(), vecSize, cudaMemcpyHostToDevice);

    MatrixVectorMultKernel<<<dimGrid, dimBlock>>>(M_d, V_d, O_d, width);

    cudaMemcpy(O_h.data(), O_d, vecSize, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(V_d);
    cudaFree(O_d);
}