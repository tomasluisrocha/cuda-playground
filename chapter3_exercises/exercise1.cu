#include <iostream>
#include <vector>
#include <cuda_runtime.h>

constexpr int THREADS_PER_BLOCK = 256;

// A
__global__ void matrixMultRowKernel(float *M, float *N, float *P, size_t width)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < width)
    {
        for (int col = 0; col < width; col++)
        {
            float Pval = 0;
            for (int i = 0; i < width; i++)
            {
                Pval += M[(row * width) + i] * N[(i * width) + col];
            }
            P[row * width + col] = Pval;
        }
    }
}

void matrixMultRow(const std::vector<float> &M_h, const std::vector<float> &N_h, std::vector<float> &P_h, size_t width)
{
    dim3 dimBlock(THREADS_PER_BLOCK);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x);
    size_t matSize = width * width;
    float *M_d, *N_d, *P_d;

    cudaMalloc((void **)&M_d, matSize * sizeof(float));
    cudaMalloc((void **)&N_d, matSize * sizeof(float));
    cudaMalloc((void **)&P_d, matSize * sizeof(float));

    cudaMemcpy(M_d, M_h.data(), matSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h.data(), matSize * sizeof(float), cudaMemcpyHostToDevice);

    matrixMultRowKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P_h.data(), P_d, matSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

// B
__global__ void matrixMultColKernel(float *M, float *N, float *P, size_t width)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < width)
    {
        for (int row = 0; row < width; row++)
        {
            float Pval = 0;
            for (int i = 0; i < width; i++)
            {
                Pval += M[(row * width) + i] * N[(i * width) + col];
            }
            P[row * width + col] = Pval;
        }
    }
}

void matrixMultCol(const std::vector<float> &M_h, const std::vector<float> &N_h, std::vector<float> &P_h, size_t width)
{
    dim3 dimBlock(THREADS_PER_BLOCK);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x);
    size_t matSize = width * width;
    float *M_d, *N_d, *P_d;

    cudaMalloc((void **)&M_d, matSize * sizeof(float));
    cudaMalloc((void **)&N_d, matSize * sizeof(float));
    cudaMalloc((void **)&P_d, matSize * sizeof(float));

    cudaMemcpy(M_d, M_h.data(), matSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h.data(), matSize * sizeof(float), cudaMemcpyHostToDevice);

    matrixMultColKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P_h.data(), P_d, matSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}