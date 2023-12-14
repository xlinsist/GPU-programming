#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

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

const int BATCH_SIZE = 1024;
const int BATCH_SIZE_2 = 16;
const int TOTAL = 10000;
const int TOTAL_1 = 9216;
const int INPUT_SIZE = 28;
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
__constant__ float FC1_BIAS[FC1];
__constant__ float FC2_BIAS[FC2];
__constant__ float FC3_BIAS[FC3];

void __global__ ConvProcessing1(const float *input, float *output)
{
    const int batch = blockIdx.x;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int outCol = tx;
    const int outRow = ty;

    __shared__ float output_s[F1][CONV1_OUTPUT_SIZE][CONV1_OUTPUT_SIZE];

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
                        sum += CONV1_WEIGHT[k][l][i][j] * 
                            input[batch * INPUT_SIZE * INPUT_SIZE * C1 +
                                l * INPUT_SIZE * INPUT_SIZE + 
                                    inRow * INPUT_SIZE + inCol];   
                    }
                }
            float tmp = sum + CONV1_BIAS[k];
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
                output[batch * F1 * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE + 
                    k * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE + 
                        outRow / 2 * POOL1_OUTPUT_SIZE + outCol / 2] = tmp3;
            }
        }
    }
}

void __global__ ConvProcessing2(const float *input, float *output)
{
    const int batch = blockIdx.x;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int outCol = tx;
    const int outRow = ty;
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
                        sum += CONV2_WEIGHT[k][l][i][j] * 
                            input[batch * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * C2 +
                                l * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE + 
                                    inRow * POOL1_OUTPUT_SIZE + inCol];   
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
                output[batch * F2 * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE + 
                    k * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE + 
                        outRow / 2 * POOL2_OUTPUT_SIZE + outCol / 2] = tmp3;
            }
        }
    }
}

void __global__ FCProcessing1(const float *input, const float *weight, float *output)
{
    const int batch = blockIdx.x;
    const int t = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < FC0; i++)
    {
        sum += input[batch * FC0 + i] * weight[t * FC0 + i];
    }
    sum += FC1_BIAS[t];
    if (sum < 0.0f)
        sum = 0.0f;
    output[batch * FC1 + t] = sum;
}

void __global__ FCProcessing2(const float *input, const float *weight, float *output)
{
    const int batch = blockIdx.x;
    const int t = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < FC1; i++)
    {
        sum += input[batch * FC1 + i] * weight[t * FC1 + i];
    }
    sum += FC2_BIAS[t];
    if (sum < 0.0f)
        sum = 0.0f;
    output[batch * FC2 + t] = sum;
}

void __global__ FCProcessing3(const float *input, const float *weight, int *prediction)
{
    __shared__ float output_s[FC3];
    const int batch = blockIdx.x;
    const int t = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < FC2; i++)
        sum += input[batch * FC2 + i] * weight[t * FC2 + i];
    output_s[t] = sum;

    __syncthreads();  
    if (t == 0) 
    {
        float bestResult = output_s[0];
        int bestId = 0;
        for (int i = 1; i < FC3; i++)
            if (output_s[i] > bestResult)
            {
                bestResult = output_s[i];
                bestId = i;
            }
        *(prediction + batch) = bestId;
    }
}

int main(int argc, char* argv[]) {
	std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
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

    float *weightFC1_d;
    const int weightFC1Bytes = FC0 * FC1 * sizeof(float);
    cudaMalloc((void **)&weightFC1_d, weightFC1Bytes);
    cudaMemcpy(weightFC1_d, fc1_weight.data(), weightFC1Bytes, cudaMemcpyHostToDevice);
    float *weightFC2_d;
    const int weightFC2Bytes = FC1  * FC2 * sizeof(float);
    cudaMalloc((void **)&weightFC2_d, weightFC2Bytes);
    cudaMemcpy(weightFC2_d, fc2_weight.data(), weightFC2Bytes, cudaMemcpyHostToDevice);
    float *weightFC3_d;
    const int weightFC3Bytes = FC2  * FC3 * sizeof(float);
    cudaMalloc((void **)&weightFC3_d, weightFC3Bytes);
    cudaMemcpy(weightFC3_d, fc3_weight.data(), weightFC3Bytes, cudaMemcpyHostToDevice);

    int total = images.size();
    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    
    int correct = 0;
    for (int t = 0; t < TOTAL_1; t += BATCH_SIZE) {

        // Read images.
        float *input_d;
        const int singleInputBytes = INPUT_SIZE * INPUT_SIZE * sizeof(float);
        cudaMalloc((void **)&input_d, BATCH_SIZE * singleInputBytes);
        for (int i = 0; i < BATCH_SIZE; i++)
            cudaMemcpy(input_d + i * INPUT_SIZE * INPUT_SIZE, images[t + i].data(), singleInputBytes, cudaMemcpyHostToDevice);

        // Process Conv1+Relu+Pool1.
        float *pool1Output_d;
        const int pool1OutputBytes = BATCH_SIZE * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * F1 * sizeof(float);
        cudaMalloc((void **)&pool1Output_d, pool1OutputBytes);
        const dim3 blockSize1(32, 32);
        ConvProcessing1<<<BATCH_SIZE, blockSize1>>>(input_d, pool1Output_d);

        // Process Conv1+Relu+Pool2.
        float *pool2Output_d;
        const int pool2OutputBytes = BATCH_SIZE * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE * F2 * sizeof(float);
        cudaMalloc((void **)&pool2Output_d, pool2OutputBytes);
        const dim3 blockSize2(16, 16);
        ConvProcessing2<<<BATCH_SIZE, blockSize2>>>(pool1Output_d, pool2Output_d);

        // Process FC1+Relu.
        float *fc1Output_d;
        const int fc1OutputBytes = BATCH_SIZE * FC1 * sizeof(float);
        cudaMalloc((void **)&fc1Output_d, fc1OutputBytes);
        FCProcessing1<<<BATCH_SIZE, FC1>>>(pool2Output_d, weightFC1_d, fc1Output_d);

        // Process FC2+Relu.
        float *fc2Output_d;
        const int fc2OutputBytes = BATCH_SIZE * FC2 * sizeof(float);
        cudaMalloc((void **)&fc2Output_d, fc2OutputBytes);
        FCProcessing2<<<BATCH_SIZE, FC2>>>(fc1Output_d, weightFC2_d, fc2Output_d);

        // Process FC3+Relu.
        int* fc3Output_d;
        const int fc3OutputBytes = BATCH_SIZE * sizeof(float);
        int* prediction = (int*) malloc(fc3OutputBytes);
        cudaMalloc((void**)&fc3Output_d, fc3OutputBytes);
        FCProcessing3<<<BATCH_SIZE, FC3>>>(fc2Output_d, weightFC3_d, fc3Output_d);
        cudaMemcpy(prediction, fc3Output_d, fc3OutputBytes, cudaMemcpyDeviceToHost);
        for (int i = 0; i < BATCH_SIZE; i++)
            if (prediction[i] == labels[t + i])
                correct++;

        free(prediction);
        cudaFree(input_d);
        cudaFree(pool1Output_d);
        cudaFree(pool2Output_d);
        cudaFree(fc1Output_d);
        cudaFree(fc2Output_d);
        cudaFree(fc3Output_d);
    }

    for (int t = TOTAL_1; t < TOTAL; t += BATCH_SIZE_2) {

        // Read images.
        float *input_d;
        const int singleInputBytes = INPUT_SIZE * INPUT_SIZE * sizeof(float);
        cudaMalloc((void **)&input_d, BATCH_SIZE_2 * singleInputBytes);
        for (int i = 0; i < BATCH_SIZE_2; i++)
            cudaMemcpy(input_d + i * INPUT_SIZE * INPUT_SIZE, images[t + i].data(), singleInputBytes, cudaMemcpyHostToDevice);

        // Process Conv1+Relu+Pool1.
        float *pool1Output_d;
        const int pool1OutputBytes = BATCH_SIZE_2 * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * F1 * sizeof(float);
        cudaMalloc((void **)&pool1Output_d, pool1OutputBytes);
        const dim3 blockSize1(32, 32);
        ConvProcessing1<<<BATCH_SIZE_2, blockSize1>>>(input_d, pool1Output_d);

        // Process Conv1+Relu+Pool2.
        float *pool2Output_d;
        const int pool2OutputBytes = BATCH_SIZE_2 * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE * F2 * sizeof(float);
        cudaMalloc((void **)&pool2Output_d, pool2OutputBytes);
        const dim3 blockSize2(16, 16);
        ConvProcessing2<<<BATCH_SIZE_2, blockSize2>>>(pool1Output_d, pool2Output_d);

        // Process FC1+Relu.
        float *fc1Output_d;
        const int fc1OutputBytes = BATCH_SIZE_2 * FC1 * sizeof(float);
        cudaMalloc((void **)&fc1Output_d, fc1OutputBytes);
        FCProcessing1<<<BATCH_SIZE_2, FC1>>>(pool2Output_d, weightFC1_d, fc1Output_d);

        // Process FC2+Relu.
        float *fc2Output_d;
        const int fc2OutputBytes = BATCH_SIZE_2 * FC2 * sizeof(float);
        cudaMalloc((void **)&fc2Output_d, fc2OutputBytes);
        FCProcessing2<<<BATCH_SIZE_2, FC2>>>(fc1Output_d, weightFC2_d, fc2Output_d);

        // Process FC3+Relu.
        int* fc3Output_d;
        const int fc3OutputBytes = BATCH_SIZE_2 * sizeof(float);
        int* prediction = (int*) malloc(fc3OutputBytes);
        cudaMalloc((void**)&fc3Output_d, fc3OutputBytes);
        FCProcessing3<<<BATCH_SIZE_2, FC3>>>(fc2Output_d, weightFC3_d, fc3Output_d);
        cudaMemcpy(prediction, fc3Output_d, fc3OutputBytes, cudaMemcpyDeviceToHost);
        for (int i = 0; i < BATCH_SIZE_2; i++)
            if (prediction[i] == labels[t + i])
                correct++;

        free(prediction);
        cudaFree(input_d);
        cudaFree(pool1Output_d);
        cudaFree(pool2Output_d);
        cudaFree(fc1Output_d);
        cudaFree(fc2Output_d);
        cudaFree(fc3Output_d);
    }
	
	// 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
	cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(6) << diff.count() << ":" << (double)correct/(double)total;
    
    cudaFree(weightFC1_d);
    cudaFree(weightFC2_d);
    cudaFree(weightFC3_d);
    return 0;
}
