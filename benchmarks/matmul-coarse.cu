#include "error.cuh"
#include <math.h>
#include <stdio.h>

const float EPSILON = 1.0e-6f;
const int NUM_REPEATS = 10;

// Change parameters based on applications.
const int TILE_NUM = 32;
const int COARSE_FACTOR = 4;
const int N = 1000;
const int length = N * N;
const float X_INIT_VALUE = 1.0f;
const float Y_INIT_VALUE = 2.0f;
const float Z_RESULT_VALUE = 2000.0f;
void check(const float *z, const int N);

/*
What tiling strategy has done:
- Correspond each sub-matrix in matrix z to a block in the grid in CUDA.
- Iterate along the "band" of matrix x and matrix y in units of sub-matrix:
    - Load the elements of each sub-matrix into shared memory.
    - Calculate the result of sub-matrix After all the loading is completed.
- Do boundary check when storing the results into matrix z.
*/
void __global__ execute(const float *x, const float *y, float *z, const int N)
{
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    // The division of grids is output-centric decomposition, which means the
    // row and col are used to locate the output.
    const int row = by * blockDim.y + ty;
    const int col = bx * blockDim.x + tx;

    __shared__ float s_x[TILE_NUM][TILE_NUM];
    __shared__ float s_y[TILE_NUM][TILE_NUM];

    float sum[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; c++)
        sum[c] = 0.0f;

    for (int p = 0; p < (N + TILE_NUM - 1)/TILE_NUM; p++)
    {
        if (row < N && p * TILE_NUM + tx < N)
            s_x[ty][tx] = x[row * N + p * TILE_NUM + tx];
        else
            s_x[ty][tx] = 0;

        for (int c = 0; c < COARSE_FACTOR; c++)
        {
            if (p * TILE_NUM + ty < N && col + c * TILE_NUM < N)
                s_y[ty][tx] = y[(p * TILE_NUM + ty) * N + col + c * TILE_NUM];
            else
                s_y[ty][tx] = 0;
            __syncthreads();

            // Apply naive matrix multiplication within a block.
            for (int i = 0; i < TILE_NUM; i++)
                sum[c] += s_x[ty][i] * s_y[i][tx];
            __syncthreads();  
        }

    }
    for (int c = 0; c < COARSE_FACTOR; c++)
        if (row < N && col + c * TILE_NUM < N)
            z[row * N + col + c * TILE_NUM] = sum[c];
}

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


