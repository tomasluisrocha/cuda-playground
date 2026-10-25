#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "tgaimage.h"
#include <sstream>

__global__ void colorToGrayscaleKernel(unsigned char *Pout, unsigned char *Pin, int width, int height)
{
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (column < width && row < height)
    {
        int grayIndexOffset = row * width + column; // {luminance}
        int colorIndexOffset = grayIndexOffset * 3; // {red, green, blue}
        unsigned char b = Pin[colorIndexOffset];
        unsigned char g = Pin[colorIndexOffset + 1];
        unsigned char r = Pin[colorIndexOffset + 2];
        Pout[grayIndexOffset] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void colorToGrayscale(const TGAImage &color_image, TGAImage &gray_image)
{
    int width = color_image.width();
    int height = color_image.height();
    gray_image = TGAImage(width, height, TGAImage::GRAYSCALE);

    int nPixel = width * height;

    unsigned char *Pin_d, *Pout_d;

    cudaMalloc((void **)&Pin_d, nPixel * 3);
    cudaMalloc((void **)&Pout_d, nPixel);

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

    colorToGrayscaleKernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);

    std::vector<unsigned char> Pout_h(nPixel);
    cudaMemcpy(Pout_h.data(), Pout_d, nPixel, cudaMemcpyDeviceToHost);

    cudaFree(Pin_d);
    cudaFree(Pout_d);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned char g = Pout_h[y * width + x];
            gray_image.set(x, y, TGAColor{g});
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

        TGAImage gray_image;
        colorToGrayscale(color_image, gray_image);

        size_t last_slash = filepath.find_last_of("/\\");
        std::string out_filename;

        if (last_slash == std::string::npos)
        {
            out_filename = "gray_" + filepath;
        }
        else
        {
            std::string dir = filepath.substr(0, last_slash + 1);
            std::string file = filepath.substr(last_slash + 1);
            out_filename = dir + "gray_" + file;
        }

        gray_image.write_tga_file(out_filename, false, false);
    }

    return 0;
}