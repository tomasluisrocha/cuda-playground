# CUDA Learning

Small CUDA programs written while working through *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj). Each program is my own implementation of an exercise from the book, with a CPU version alongside where I wanted to compare timings.

This is a work in progress. I am going through the book in order.

## Contents

| Folder | Program | What it does | Book topic |
|---|---|---|---|
| `vector/` | `vector.c`, `vector.cu` | Vector addition on the CPU (C) and on the GPU (CUDA) | Data parallel computing |
| `grayscaler/` | `grayscaler.cu` | RGB to grayscale on the GPU, one thread per pixel | Multidimensional grids and data |
| `grayscaler/` | `grayscaler_cpu.cpp` | The same conversion on the CPU, with per-stage timing | Baseline for the GPU version |
| `blur/` | `blur.cu` | Box blur on the GPU (3x3 window by default) | Multidimensional grids and data |
| `matrix/` | `matrixmult.cu` | Square matrix multiplication on the GPU | Matrix Multiplication 

## Requirements

- An NVIDIA GPU and the CUDA Toolkit (`nvcc`)
- A C/C++ compiler (`gcc`/`g++`)
- Input images must be **24-bit TGA** files. The image programs assume three channels per pixel.

## Build and run

The programs read images through a small TGA library (`tgaimage.h`, `tgaimage.cpp`), so it is compiled together with each program.

**Vector addition**

```bash
cd vector
gcc -O2 vector.c -o vector_c
nvcc -O2 vector.cu -o vector_cuda
```

**Grayscale**

```bash
cd grayscaler
nvcc -O2 grayscaler.cu tgaimage.cpp -o grayscaler
g++ -O2 -fopenmp grayscaler_cpu.cpp tgaimage.cpp -o grayscaler_cpu

./grayscaler images/lenna_512.tga        # writes images/gray_lenna_512.tga
./grayscaler_cpu images/lenna_512.tga    # writes images/gray_cpu_lenna_512.tga and prints timings
```

**Blur**

```bash
cd blur
nvcc -O2 blur.cu tgaimage.cpp -o blur

./blur images/lenna_512.tga              # writes images/blur_lenna_512.tga
```

The image programs accept several input files in one run. Output files are written next to their inputs with a `gray_`, `gray_cpu_` or `blur_` prefix.

The blur window is set by the `BLUR_SIZE` constant in `blur.cu`. A value of `n` gives a `(2n+1)` box blur. Pixels at the border average only the neighbours that exist.

The `-fopenmp` flag in the CPU build is optional. Without it the conversion runs on one core.

**Matrix multiplication**

```bash
cd matrixmul
nvcc -O3 matrixmul.cu -o matrixmul

./matrixmul
```

Or pipe inputs directly:

```bash
echo "3
1.0 2.9 1 2 3 2 7 8 9
9.0 2 3 9 8 7 4 9 6" | ./matrixmul
```

The program calculates \(P = M \times N\) for square matrices of dimensions \(n \times n\), using a one-thread-per-element grid mapping.

**Input format:**

1. Matrix dimension \(n\)
2. \(n^2\) values for matrix \(M\), in row-major order
3. \(n^2\) values for matrix \(N\), in row-major order

## Credits

- The exercises follow *Programming Massively Parallel Processors: A Hands-on Approach* by Wen-mei W. Hwu, David B. Kirk and Izzat El Hajj. The code here is my own implementation.
- `tgaimage.h` and `tgaimage.cpp` come from the [Tiny Renderer](https://github.com/ssloy/tinyrenderer) course by Dmitry V. Sokolov and are used under its permissive license.