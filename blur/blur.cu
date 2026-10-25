#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "tgaimage.h"
#include <sstream>

constexpr int BLUR_SIZE = 3;

__global__ void blurKernel(unsigned char *out, unsigned char *in, int width, int height)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col < width && row < height)
    {
        int pixValB = 0;
        int pixValG = 0;
        int pixValR = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
        {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curCol >= 0 && curRow < height && curCol < width)
                {
                    int curPixelIdx = (curRow * width + curCol) * 3;
                    pixValB += in[curPixelIdx];
                    pixValG += in[curPixelIdx + 1];
                    pixValR += in[curPixelIdx + 2];
                    ++pixels;
                }
            }
        }
        int outPixelIdx = (row * width + col) * 3;
        out[outPixelIdx + 0] = (unsigned char)(pixValB / pixels);
        out[outPixelIdx + 1] = (unsigned char)(pixValG / pixels);
        out[outPixelIdx + 2] = (unsigned char)(pixValR / pixels);
    }
}

void blur(const TGAImage &color_image, TGAImage &blur_image)
{
    int width = color_image.width();
    int height = color_image.height();
    blur_image = TGAImage(width, height, TGAImage::RGB);

    int nPixel = width * height;

    unsigned char *Pin_d, *Pout_d;

    cudaMalloc((void **)&Pin_d, nPixel * 3);
    cudaMalloc((void **)&Pout_d, nPixel * 3);

    std::vector<unsigned char> Pin_h(nPixel * 3);
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            TGAColor c = color_image.get(x, y);
            int pixel_offset = (y * width + x) * 3;
            Pin_h[pixel_offset] = c[0];
            Pin_h[pixel_offset + 1] = c[1];
            Pin_h[pixel_offset + 2] = c[2];
        }
    }

    cudaMemcpy(Pin_d, Pin_h.data(), nPixel * 3, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0));
    dim3 dimBlock(16, 16);

    blurKernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);

    cudaDeviceSynchronize();

    std::vector<unsigned char> Pout_h(nPixel * 3);
    cudaMemcpy(Pout_h.data(), Pout_d, nPixel * 3, cudaMemcpyDeviceToHost);

    cudaFree(Pin_d);
    cudaFree(Pout_d);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int pixelIdx = (y * width + x) * 3;
            unsigned char b = Pout_h[pixelIdx];
            unsigned char g = Pout_h[pixelIdx + 1];
            unsigned char r = Pout_h[pixelIdx + 2];

            blur_image.set(x, y, TGAColor{b, g, r});
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: ./program path_to_images" << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; i++)
    {
        // Get filepath
        std::string filepath = argv[i];

        // Initialize color image
        TGAImage color_image;
        color_image.read_tga_file(filepath);

        TGAImage blur_image;
        blur(color_image, blur_image);

        size_t last_slash = filepath.find_last_of("/\\");
        std::string out_filename;

        if (last_slash == std::string::npos)
        {
            out_filename = "blur_" + filepath;
        }
        else
        {
            std::string dir = filepath.substr(0, last_slash + 1);
            std::string file = filepath.substr(last_slash + 1);
            out_filename = dir + "blur_" + file;
        }

        blur_image.write_tga_file(out_filename, false, false);
    }

    return 0;
}