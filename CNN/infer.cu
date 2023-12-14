// unoptimized baseline: 1.465481:0.897000

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include "error.cuh"

const int TOTAL = 10000;
const int INPUT_SIZE = 28;
// const int CLASS_NUM = 10;
double images_arr[TOTAL][INPUT_SIZE*INPUT_SIZE];

// 读取MNIST数据集
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_images = ((num_images & 0xff000000) >> 24) | ((num_images & 0x00ff0000) >> 8) |
                 ((num_images & 0x0000ff00) << 8) | ((num_images & 0x000000ff) << 24);
    num_rows = ((num_rows & 0xff000000) >> 24) | ((num_rows & 0x00ff0000) >> 8) |
                ((num_rows & 0x0000ff00) << 8) | ((num_rows & 0x000000ff) << 24);
    num_cols = ((num_cols & 0xff000000) >> 24) | ((num_cols & 0x00ff0000) >> 8) |
                ((num_cols & 0x0000ff00) << 8) | ((num_cols & 0x000000ff) << 24);

    int image_size = num_rows * num_cols;
    std::vector<std::vector<float>> images(num_images, std::vector<float>(image_size));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            // Apply normalization based on the given mean and std.
            images[i][j] = (static_cast<float>(pixel) / 255.0f - 0.5f) / 0.5f;
        }
    }
    return images;
}

// 读取MNIST label数据集
std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_items = ((num_items & 0xff000000) >> 24) | ((num_items & 0x00ff0000) >> 8) |
                ((num_items & 0x0000ff00) << 8) | ((num_items & 0x000000ff) << 24);

    std::vector<int> labels(num_items);
    for (int i = 0; i < num_items; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }
    return labels;
}

// 读取模型参数
std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    // std::cout << "Path: " << path << std::endl;
    std::vector<float> params;
    float param;
    while (file >> param) {
        params.push_back(param);
    }
    return params;
}

const int KERNEL_SIZE = 5;
const int CONV1_OUTPUT_SIZE = 24;
const int POOL1_OUTPUT_SIZE = 12;
const int CONV2_OUTPUT_SIZE = 8;
const int POOL2_OUTPUT_SIZE = 4;
const int F1 = 6;
const int F2 = 16;
const int C1 = 1;
const int C2 = 6;
const int FC0 = 256;
const int FC1 = 120;
const int FC2 = 84;
const int FC3 = 10;

// Declare small-size weight and bias in constant memory.
__constant__ float CONV1_WEIGHT[F1][C1][KERNEL_SIZE][KERNEL_SIZE];
__constant__ float CONV2_WEIGHT[F2][C2][KERNEL_SIZE][KERNEL_SIZE];
__constant__ float CONV1_BIAS[F1];
__constant__ float CONV2_BIAS[F2];
// __constant__ float FC2_WEIGHT[FC1][FC2];
// __constant__ float FC3_WEIGHT[FC2][FC3];
__constant__ float FC1_BIAS[FC1];
__constant__ float FC2_BIAS[FC2];
__constant__ float FC3_BIAS[FC3];

void __global__ ConvProcessing1(const float *input, float *output)
{
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int outCol = bx * blockDim.x + tx;
    const int outRow = by * blockDim.y + ty;
    __shared__ float output_s[F1][CONV1_OUTPUT_SIZE][CONV1_OUTPUT_SIZE];

    // if (outCol >= 0 && outCol < CONV1_OUTPUT_SIZE && outRow >= 0 && outRow < CONV1_OUTPUT_SIZE)
    //     for (int k = 0; k < F1; k++)
    //     {
    //         output[k * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE + outRow * CONV1_OUTPUT_SIZE + outCol] = 1.0f;
    //         printf("debug: %d %d %d %.6f\n", k, outRow, outCol, 
    //             output[k * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE + outRow * CONV1_OUTPUT_SIZE + outCol]);
    //     }
    
    // The thread is bound to the output of conv1.
    if (outCol < CONV1_OUTPUT_SIZE && outRow < CONV1_OUTPUT_SIZE)
    {
        for (int k = 0; k < F1; k++)
        {    
            float sum = 0.0f;
            for (int i = 0; i < KERNEL_SIZE; i++)
                for (int j = 0; j < KERNEL_SIZE; j++)
                {
                    int inCol = outCol + j;
                    int inRow = outRow + i;
                    // Since the input size and output size are fixed and given, there is no need to check boundary here.
                    for (int l = 0; l < C1; l++)
                    {
                        sum += CONV1_WEIGHT[k][l][i][j] * input[l * INPUT_SIZE * INPUT_SIZE + inRow * INPUT_SIZE + inCol];   
                        // printf("debug1: %d %d %.6f\n", outRow, outCol, sum);
                    }
                }
            // printf("debug2: %d %d %.6f\n", outRow, outCol, sum);  
            float tmp = sum + CONV1_BIAS[k];
            if (tmp < 0.0f)
                tmp = 0.0f;
            output_s[k][outRow][outCol] = tmp;
            __syncthreads();

            if ((outCol & 1) && (outRow & 1))
            {
                // printf("debug2: %d %d %d\n", outRow, outCol,
                // k * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE + outRow / 2 * POOL1_OUTPUT_SIZE + outCol / 2);
                float tmp1 = output_s[k][outRow][outCol] > output_s[k][outRow][outCol - 1] ? 
                    output_s[k][outRow][outCol] : output_s[k][outRow][outCol - 1];
                float tmp2 = output_s[k][outRow - 1][outCol] > output_s[k][outRow - 1][outCol - 1] ? 
                    output_s[k][outRow - 1][outCol] : output_s[k][outRow - 1][outCol - 1];
                float tmp3 = tmp1 > tmp2 ? tmp1 : tmp2;
                output[k * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE + outRow / 2 * POOL1_OUTPUT_SIZE + outCol / 2] = tmp3;
            }
        }
    }
}

void __global__ ConvProcessing2(const float *input, float *output)
{
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int outCol = bx * blockDim.x + tx;
    const int outRow = by * blockDim.y + ty;
    __shared__ float output_s[F2][CONV2_OUTPUT_SIZE][CONV2_OUTPUT_SIZE];
            
    if (outCol < CONV2_OUTPUT_SIZE && outRow < CONV2_OUTPUT_SIZE)
    {
        for (int k = 0; k < F2; k++)
        {    
            float sum = 0.0f;
            for (int i = 0; i < KERNEL_SIZE; i++)
                for (int j = 0; j < KERNEL_SIZE; j++)
                {
                    int inCol = outCol + j;
                    int inRow = outRow + i;
                    for (int l = 0; l < C2; l++)
                    {
                        sum += CONV2_WEIGHT[k][l][i][j] * input[l * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE + inRow * POOL1_OUTPUT_SIZE + inCol];   
                    }
                }
            float tmp = sum + CONV2_BIAS[k];
            if (tmp < 0.0f)
                tmp = 0.0f;
            output_s[k][outRow][outCol] = tmp;
            __syncthreads();

            if ((outCol & 1) && (outRow & 1))
            {
                float tmp1 = output_s[k][outRow][outCol] > output_s[k][outRow][outCol - 1] ? 
                    output_s[k][outRow][outCol] : output_s[k][outRow][outCol - 1];
                float tmp2 = output_s[k][outRow - 1][outCol] > output_s[k][outRow - 1][outCol - 1] ? 
                    output_s[k][outRow - 1][outCol] : output_s[k][outRow - 1][outCol - 1];
                float tmp3 = tmp1 > tmp2 ? tmp1 : tmp2;
                output[k * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE + outRow / 2 * POOL2_OUTPUT_SIZE + outCol / 2] = tmp3;
                // printf("debug %d %d %d %.6f\n", k, outRow / 2, outCol / 2, tmp3);
            }
        }
    }
}

void __global__ FCProcessing1(const float *input, const float *weight, float *output)
{
    const int t = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < FC0; i++)
    {
        sum += input[i] * weight[t * FC0 + i];
        // if (t==119)
        //     printf("debug %d %.6f %.6f\n", i, input[i], weight[t * FC0 + i]);
    }
    sum += FC1_BIAS[t];
    if (sum < 0.0f)
        sum = 0.0f;
    output[t] = sum;
}

void __global__ FCProcessing2(const float *input, const float *weight, float *output)
{
    const int t = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < FC1; i++)
    {
        sum += input[i] * weight[t * FC1 + i];
    }
    sum += FC2_BIAS[t];
    if (sum < 0.0f)
        sum = 0.0f;
    output[t] = sum;
}

void __global__ FCProcessing3(const float *input, const float *weight, float *output)
{
    const int t = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < FC2; i++)
    {
        sum += input[i] * weight[t * FC2 + i];
    }
    output[t] = sum;
}

// void __global__ FCProcessing3(const float *input, const float *weight, int &prediction)
// {
//     const int t = threadIdx.x;
//     float sum = 0.0f;
//     for (int i = 0; i < FC2; i++)
//     {
//         sum += input[i] * weight[i * FC3 + t];
//     }
//     __shared__ float output_s[FC3];
//     if (t < FC3)
//         output_s[t] = sum;
//     __syncthreads();
//     printf("debug %d %.6f\n", t, sum);

//     float bestResult = output_s[0];
//     int bestId = 0;
//     if (t == 0) 
//     {
//     //     for (int i = 1; i < FC3; i++)
//     //         if (output_s[i] > bestResult)
//     //         {
//     //             bestResult = output_s[i];
//     //             bestId = i;
//     //         }
//     }
//     prediction = 0;
//     printf("debug %d %d\n", t, prediction);
// }

int main(int argc, char* argv[]) {
	// std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
    std::string dir = "/home/zhouxulin/CNN";
    // std::cout << dir;
    // 读取测试集，对于想实现CUDA C/C++训练的同学，参考训练集文件名为train-images-idx3-ubyte和train-labels-idx1-ubyte
    auto images = read_mnist_images(dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    // 读取测试集标签
    auto labels = read_mnist_labels(dir + "/../../data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    // 读取模型参数
    auto conv1_weight = read_param(dir + "/conv1.weight.txt");
    auto conv1_bias = read_param(dir + "/conv1.bias.txt");
    auto conv2_weight = read_param(dir + "/conv2.weight.txt");
    auto conv2_bias = read_param(dir + "/conv2.bias.txt");
    auto fc1_weight = read_param(dir + "/fc1.weight.txt");
    auto fc1_bias = read_param(dir + "/fc1.bias.txt");
    auto fc2_weight = read_param(dir + "/fc2.weight.txt");
    auto fc2_bias = read_param(dir + "/fc2.bias.txt");
    auto fc3_weight = read_param(dir + "/fc3.weight.txt");
    auto fc3_bias = read_param(dir + "/fc3.bias.txt");
    cudaMemcpyToSymbol(CONV1_BIAS, conv1_bias.data(), F1 * sizeof(float));
    cudaMemcpyToSymbol(CONV2_BIAS, conv2_bias.data(), F2 * sizeof(float));
    cudaMemcpyToSymbol(CONV1_WEIGHT, conv1_weight.data(), F1 * C1 * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMemcpyToSymbol(CONV2_WEIGHT, conv2_weight.data(), F2 * C2 * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMemcpyToSymbol(FC1_BIAS, fc1_bias.data(), FC1 * sizeof(float));
    cudaMemcpyToSymbol(FC2_BIAS, fc2_bias.data(), FC2 * sizeof(float));
    cudaMemcpyToSymbol(FC3_BIAS, fc3_bias.data(), FC3 * sizeof(float));
    // cudaMemcpyToSymbol(FC1_WEIGHT, fc1_weight.data(), ... * sizeof(float));
    // cudaMemcpyToSymbol(FC2_WEIGHT, fc2_weight.data(), FC1 * FC2 * sizeof(float));
    // cudaMemcpyToSymbol(FC3_WEIGHT, fc3_weight.data(), FC2 * FC3 * sizeof(float));
    float *weightFC1_d;
    const int weightFC1Bytes = FC0 * FC1 * sizeof(float);
    CHECK(cudaMalloc((void **)&weightFC1_d, weightFC1Bytes));
    CHECK(cudaMemcpy(weightFC1_d, fc1_weight.data(), weightFC1Bytes, cudaMemcpyHostToDevice));
    float *weightFC2_d;
    const int weightFC2Bytes = FC1  * FC2 * sizeof(float);
    CHECK(cudaMalloc((void **)&weightFC2_d, weightFC2Bytes));
    CHECK(cudaMemcpy(weightFC2_d, fc2_weight.data(), weightFC2Bytes, cudaMemcpyHostToDevice));
    float *weightFC3_d;
    const int weightFC3Bytes = FC2  * FC3 * sizeof(float);
    CHECK(cudaMalloc((void **)&weightFC3_d, weightFC3Bytes));
    CHECK(cudaMemcpy(weightFC3_d, fc3_weight.data(), weightFC3Bytes, cudaMemcpyHostToDevice));

    int total = images.size();
    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();

    // 进行推理
	// std::cout << images[0].size() << std::endl;
	
	// 参数加载
	// std::cout << fc3_bias.size() << std::endl;
    
    int correct = 0;
    for (int t = 0; t < TOTAL; t++) {

        // Read images.
        float *input_d;
        const int inputBytes = INPUT_SIZE * INPUT_SIZE * sizeof(float);
        CHECK(cudaMalloc((void **)&input_d, inputBytes));
        CHECK(cudaMemcpy(input_d, images[t].data(), inputBytes, cudaMemcpyHostToDevice));

        // Process Conv1+Relu+Pool1.
        float *pool1Output_d;
        const int pool1OutputBytes = POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * F1 * sizeof(float);
        CHECK(cudaMalloc((void **)&pool1Output_d, pool1OutputBytes));
        const dim3 blockSize1(32, 32);
        const dim3 gridSize1(1, 1);
        ConvProcessing1<<<gridSize1, blockSize1>>>(input_d, pool1Output_d);
        // float *pool1Output_h = (float*) malloc(pool1OutputBytes);
        // CHECK(cudaMemcpy(pool1Output_h, pool1Output_d, pool1OutputBytes, cudaMemcpyDeviceToHost));
        // for (int i=0;i<F1;i++)
        // {
        //     printf("[");
        //     for (int j=0;j<POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;j++) 
        //     {
        //         printf("%.6f ",pool1Output_h[k*F1*POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE + i*POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE + j]);   
        //         if (j%4==3) printf("\n");
        //     }
        //     printf("]\n");
        // }

        // Process Conv1+Relu+Pool2.
        float *pool2Output_d;
        const int pool2OutputBytes = POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE * F2 * sizeof(float);
        CHECK(cudaMalloc((void **)&pool2Output_d, pool2OutputBytes));
        const dim3 blockSize2(16, 16);
        const dim3 gridSize2(1, 1);
        ConvProcessing2<<<gridSize2, blockSize2>>>(pool1Output_d, pool2Output_d);
        // float *pool2Output_h = (float*) malloc(pool2OutputBytes);
        // CHECK(cudaMemcpy(pool2Output_h, pool2Output_d, pool2OutputBytes, cudaMemcpyDeviceToHost));
        // printf("debug in host: %.6f %.6f\n",pool2Output_h[0], pool2Output_h[100]);
        // for (int i=0;i<F2;i++)
        // {
        //     printf("[");
        //     for (int j=0;j<POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE;j++) 
        //     {
        //         printf("%.6f ",pool2Output_h[k*F2*POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE + i*POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE + j]);   
        //         if (j%4==3) printf("\n");
        //     }
        //     printf("]\n");
        // }
        
        // Process FC1+Relu.
        float *fc1Output_d;
        const int fc1OutputBytes = FC1 * sizeof(float);
        CHECK(cudaMalloc((void **)&fc1Output_d, fc1OutputBytes));
        FCProcessing1<<<1, FC1>>>(pool2Output_d, weightFC1_d, fc1Output_d);
        float *fc1Output_h = (float*) malloc(fc1OutputBytes);
        CHECK(cudaMemcpy(fc1Output_h, fc1Output_d, fc1OutputBytes, cudaMemcpyDeviceToHost));
        // for (int i=0;i<FC1;i++) 
        // {
        //     printf("%.6f ",fc1Output_h[i]);
        //     if (i%6==5) printf("\n");
        // }

        // Process FC2+Relu.
        float *fc2Output_d;
        const int fc2OutputBytes = FC2 * sizeof(float);
        CHECK(cudaMalloc((void **)&fc2Output_d, fc2OutputBytes));
        FCProcessing2<<<1, FC2>>>(fc1Output_d, weightFC2_d, fc2Output_d);
        float *fc2Output_h = (float*) malloc(fc2OutputBytes);
        CHECK(cudaMemcpy(fc2Output_h, fc2Output_d, fc2OutputBytes, cudaMemcpyDeviceToHost));
        // for (int i=0;i<FC2;i++) 
        // {
        //     printf("%.6f ",fc2Output_h[i]);
        //     if (i%6==5) printf("\n");
        // }

        // Process FC3+Relu.
        float *fc3Output_d;
        const int fc3OutputBytes = FC3 * sizeof(float);
        CHECK(cudaMalloc((void **)&fc3Output_d, fc3OutputBytes));
        FCProcessing3<<<1, FC3>>>(fc2Output_d, weightFC3_d, fc3Output_d);
        float *fc3Output_h = (float*) malloc(fc3OutputBytes);
        CHECK(cudaMemcpy(fc3Output_h, fc3Output_d, fc3OutputBytes, cudaMemcpyDeviceToHost));
        // for (int i=0;i<FC3;i++) printf("debug %d %d %.6f\n", t, i, fc3Output_h[i]);

        float bestResult = fc3Output_h[0];
        int bestId = 0;
        for (int i=1;i<FC3;i++) 
            if (bestResult < fc3Output_h[i])
            {
                bestResult = fc3Output_h[i];
                bestId = i;
            }
        // printf("debug %d %d %d\n", t, labels[t], bestId);
        if (bestId == labels[t])
            correct++;

        // Process FC3+Relu.
        // int prediction = 0;
        // FCProcessing3<<<16, FC3>>>(fc2Output_d, weightFC3_d, prediction);
        // if (prediction == labels[t])
        //     correct++;
        free(fc3Output_h);
        CHECK(cudaFree(input_d));
        CHECK(cudaFree(pool1Output_d));
        CHECK(cudaFree(pool2Output_d));
        CHECK(cudaFree(fc1Output_d));
        CHECK(cudaFree(fc2Output_d));
    }
	
	// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
	cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(6) << diff.count() << ":" << (double)correct/(double)total;
    
    CHECK(cudaFree(weightFC1_d));
    CHECK(cudaFree(weightFC2_d));
    CHECK(cudaFree(weightFC3_d));
    return 0;
}
