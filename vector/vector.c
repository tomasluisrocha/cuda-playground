#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vectorAdd(float *A, float *B, float *C, int n)
{
    for (int i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
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