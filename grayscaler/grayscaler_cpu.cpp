#include <chrono>
#include <iostream>
#include <string>
#include "tgaimage.h"

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}


void colorToGrayscaleCPU(const TGAImage &color_image, TGAImage &gray_image)
{
    int width = color_image.width();
    int height = color_image.height();
    gray_image = TGAImage(width, height, TGAImage::GRAYSCALE);

    // Pixels are independent, so rows can be split across cores.
    // Without -fopenmp this pragma is ignored and the loop runs on one core.
#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            TGAColor c = color_image.get(x, y);
            unsigned char b = c[0];
            unsigned char g = c[1];
            unsigned char r = c[2];
            unsigned char gray = 0.299f * r + 0.587f * g + 0.114f * b;
            gray_image.set(x, y, TGAColor{gray});
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
        std::string filepath = argv[i];

        Clock::time_point t_start = Clock::now();

        // Read
        Clock::time_point t0 = Clock::now();
        TGAImage color_image;
        if (!color_image.read_tga_file(filepath))
        {
            std::cerr << "skipping " << filepath << " (could not read it)" << std::endl;
            continue;
        }
        double t_read = ms_since(t0);

        Clock::time_point t1 = Clock::now();
        TGAImage gray_image;
        colorToGrayscaleCPU(color_image, gray_image);
        double t_convert = ms_since(t1);

        size_t last_slash = filepath.find_last_of("/\\");
        std::string out_filename;
        if (last_slash == std::string::npos)
        {
            out_filename = "gray_cpu_" + filepath;
        }
        else
        {
            std::string dir = filepath.substr(0, last_slash + 1);
            std::string file = filepath.substr(last_slash + 1);
            out_filename = dir + "gray_cpu_" + file;
        }

        Clock::time_point t2 = Clock::now();
        gray_image.write_tga_file(out_filename, false, false);
        double t_write = ms_since(t2);

        std::cout << filepath << ": read " << t_read << " ms | convert " << t_convert
                  << " ms | write " << t_write << " ms | total " << ms_since(t_start) << " ms"
                  << std::endl;
    }

    return 0;
}