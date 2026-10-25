#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

constexpr int BLOCK_WIDTH = 16;

// Only handles square matrices
__global__ void matrixMultKernel(float *M, float *N, float *P, int width)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if ((col < width) && (row < width))
    {
        float Pval = 0;
        for (int i = 0; i < width; i++)
        {
            Pval += M[(row * width) + i] * N[(i * width) + col];
        }
        P[row * width + col] = Pval;
    }
}

void matrixMult(const std::vector<float> &M_h, const std::vector<float> &N_h, std::vector<float> &P_h, int width)
{
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid(ceil((float)width / dimBlock.x), ceil((float)width / dimBlock.y));

    size_t matSize = width * width;
    float *M_d, *N_d, *P_d;

    cudaMalloc((void **)&M_d, matSize * sizeof(float));
    cudaMalloc((void **)&N_d, matSize * sizeof(float));
    cudaMalloc((void **)&P_d, matSize * sizeof(float));

    cudaMemcpy(M_d, M_h.data(), matSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h.data(), matSize * sizeof(float), cudaMemcpyHostToDevice);

    matrixMultKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P_h.data(), P_d, matSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main()
{
    int width = 0;
    std::cout << "Enter dimension width: ";
    if (!(std::cin >> width) || width <= 0) return 1;

    size_t numElements = static_cast<size_t>(width) * width;

    std::vector<float> M_h(numElements);
    std::vector<float> N_h(numElements);
    std::vector<float> P_h(numElements, 0.0f);

    std::cout << "Enter M (" << numElements << " elements): ";
    for (size_t i = 0; i < numElements; ++i) std::cin >> M_h[i];

    std::cout << "Enter N (" << numElements << " elements): ";
    for (size_t i = 0; i < numElements; ++i) std::cin >> N_h[i];

    matrixMult(M_h, N_h, P_h, width);

    std::cout << "\nResult Matrix P (" << width << "x" << width << "):\n";
    std::cout << std::fixed << std::setprecision(2);
    for (int r = 0; r < width; ++r)
    {
        for (int c = 0; c < width; ++c)
        {
            std::cout << std::setw(8) << P_h[r * width + c] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}