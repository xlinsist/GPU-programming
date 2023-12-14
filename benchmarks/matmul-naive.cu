#include "error.cuh"
#include <math.h>
#include <stdio.h>

const float EPSILON = 1.0e-6f;
const int NUM_REPEATS = 10;

// Change parameters based on applications.
const int N = 1000;
const int length = N * N;
const float X_INIT_VALUE = 1.0f;
const float Y_INIT_VALUE = 2.0f;
const float Z_RESULT_VALUE = 2000.0f;
void __global__ execute(const float *x, const float *y, float *z, const int N);
void check(const float *z, const int N);

int main(void)
{
    const int numBytes = sizeof(float) * length;
    float *h_x = (float*) malloc(numBytes);
    float *h_y = (float*) malloc(numBytes);
    float *h_z = (float*) malloc(numBytes);

    for (int i = 0; i < length; ++i)
    {
        h_x[i] = X_INIT_VALUE;
        h_y[i] = Y_INIT_VALUE;
    }

    float *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **)&d_x, numBytes));
    CHECK(cudaMalloc((void **)&d_y, numBytes));
    CHECK(cudaMalloc((void **)&d_z, numBytes));
    CHECK(cudaMemcpy(d_x, h_x, numBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, numBytes, cudaMemcpyHostToDevice));

    // Define the grid and block dimensions.
    const dim3 blockSize(32, 32);
    const dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

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
        execute<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);

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

    CHECK(cudaMemcpy(h_z, d_z, numBytes, cudaMemcpyDeviceToHost));
    check(h_z, length);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

void __global__ execute(const float *x, const float *y, float *z, const int N)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (row < N && col < N) 
    {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += x[row * N + i] * y[i * N + col];
        }
        z[row * N + col] = sum;
    }
}

void check(const float *z, const int N)
{
    bool has_error = false;
    for (int i = 0; i < N; ++i)
    {
        if (fabs(z[i] - Z_RESULT_VALUE) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
    if (has_error) 
    {
        printf("The result should be %.f, but got %.f.", Z_RESULT_VALUE, z[0]);
    }
}


