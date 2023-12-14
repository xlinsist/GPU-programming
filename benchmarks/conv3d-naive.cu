#include "error.cuh"
#include <math.h>
#include <stdio.h>

const float EPSILON = 1.0e-6f;
const int NUM_REPEATS = 10;

// Change parameters based on applications.
const int C = 6;
const int F = 16;
const int INPUT_SIZE = 28;
const int KERNEL_SIZE = 5;
const int OUTPUT_SIZE = 24;
const float INPUT_INIT_VALUE = 2.0f;
const float KERNEL_INIT_VALUE = 1.0f;
const float OUTPUT_RESULT_VALUE = 300.0f;
void check(const float *z, const int N);

// Declare kernel arrray in constant memory.
__constant__ float KERNEL[F][KERNEL_SIZE][KERNEL_SIZE];
void __global__ execute(const float *input, float *output, const int n, const int c, const int f)
{
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int outCol = bx * blockDim.x + tx;
    const int outRow = by * blockDim.y + ty;

    for (int k = 0; k < f; k++)
    {    
        float sum = 0.0f;
        for (int i = 0; i < KERNEL_SIZE; i++)
            for (int j = 0; j < KERNEL_SIZE; j++)
            {
                int inCol = outCol + j;
                int inRow = outRow + i;
                if (inCol >= 0 && inCol < n && inRow >= 0 && inRow < n)
                    for (int l = 0; l < c; l++)
                        sum += KERNEL[k][i][j] * input[l * n * n + inRow * n + inCol];
            }
        int outputSize = n - KERNEL_SIZE + 1;
        if (outCol >= 0 && outCol < n - KERNEL_SIZE + 1 && outRow >= 0 && outRow < n - KERNEL_SIZE + 1)
            output[k * outputSize * outputSize + outRow * outputSize + outCol] = sum;
    }
    
}

int main(void)
{
    const int numInputBytes = sizeof(float) * C * INPUT_SIZE * INPUT_SIZE;
    const int numKernelBytes = sizeof(float) * F * KERNEL_SIZE * KERNEL_SIZE;
    const int numOutputBytes = sizeof(float) * F * OUTPUT_SIZE * OUTPUT_SIZE;
    float *h_x = (float*) malloc(numInputBytes);
    float *h_y = (float*) malloc(numKernelBytes);
    float *h_z = (float*) malloc(numOutputBytes);

    for (int i = 0; i < C * INPUT_SIZE * INPUT_SIZE; ++i)
        h_x[i] = INPUT_INIT_VALUE;
    for (int i = 0; i < F * KERNEL_SIZE * KERNEL_SIZE; ++i)
        h_y[i] = KERNEL_INIT_VALUE;

    float *d_x, *d_z;
    CHECK(cudaMalloc((void **)&d_x, numInputBytes));
    CHECK(cudaMalloc((void **)&d_z, numOutputBytes));
    CHECK(cudaMemcpy(d_x, h_x, numInputBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(KERNEL, h_y, numKernelBytes));

    // Define the grid and block dimensions.
    const dim3 blockSize(16, 16);
    const dim3 gridSize((OUTPUT_SIZE + blockSize.x - 1) / blockSize.x, (OUTPUT_SIZE + blockSize.y - 1) / blockSize.y);

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 1; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        // Execute main function here.
        execute<<<gridSize, blockSize>>>(d_x, d_z, INPUT_SIZE, C, F);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsedTime;
        CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        printf("Time = %g ms.\n", elapsedTime);
        t_sum += elapsedTime;
        t2_sum += elapsedTime * elapsedTime;
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    CHECK(cudaMemcpy(h_z, d_z, numOutputBytes, cudaMemcpyDeviceToHost));
    check(h_z, OUTPUT_SIZE * OUTPUT_SIZE);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_z));
    return 0;
}

void check(const float *z, const int length)
{
    bool has_error = false;
    for (int i = 0; i < length; ++i)
    {
        if (fabs(z[i] - OUTPUT_RESULT_VALUE) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
    if (has_error) 
    {
        printf("The result should be %.f, but got %.f.\n", OUTPUT_RESULT_VALUE, z[0]);
    }
}
