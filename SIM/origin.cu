// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc -arch=sm_70 --cudart shared test.cu -o test -Xcompiler "-O3 -std=c++14"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef TIMEPROF
#include "timeprof.h"
#else
#define timeprof_start_(x)
#define timeprof_end_()
#define timeprof_print_frame_sorted_()
#endif

const int in_channel = 1;
const int in_height = 28;
const int in_weight = 28;

const int batch_size = 10000;
// const int batch_size = 3;

// conv 1
const int conv_1_kernel_size = 2;
const int conv_1_stride = 2;
const int conv_1_padding = 0;

const int conv_1_out_channel = 1;
const int conv_1_out_height =
    (in_height - conv_1_kernel_size + 2 * conv_1_padding) / conv_1_stride + 1;
const int conv_1_out_width =
    (in_weight - conv_1_kernel_size + 2 * conv_1_padding) / conv_1_stride + 1;

// Linear 1
const int linear_1_in_size = conv_1_out_channel * conv_1_out_height * conv_1_out_width;
const int linear_1_out_size = 64;

// Linear 2
const int linear_2_out_size = 10;

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorString(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

// 读取MNIST数据集
std::vector<std::vector<float>> read_mnist_images(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cout << "Cannot open file!" << std::endl;
    return {};
  }

  int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
  file.read((char *)&magic_number, sizeof(magic_number));
  file.read((char *)&num_images, sizeof(num_images));
  file.read((char *)&num_rows, sizeof(num_rows));
  file.read((char *)&num_cols, sizeof(num_cols));

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

  // printf("num_images=%d, num_rows=%d, num_cols=%d\n", num_images, num_rows, num_cols);

  for (int i = 0; i < num_images; ++i) {
    for (int j = 0; j < image_size; ++j) {
      unsigned char pixel = 0;
      file.read((char *)&pixel, sizeof(pixel));
      images[i][j] = static_cast<float>(pixel) / 255.0f;
    }
  }

  return images;
}

// 读取MNIST label数据集
std::vector<int> read_mnist_labels(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cout << "Cannot open file!" << std::endl;
    return {};
  }

  int magic_number = 0, num_items = 0;
  file.read((char *)&magic_number, sizeof(magic_number));
  file.read((char *)&num_items, sizeof(num_items));

  // Reverse Integers (MNIST data is in big endian format)
  magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                 ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
  num_items = ((num_items & 0xff000000) >> 24) | ((num_items & 0x00ff0000) >> 8) |
              ((num_items & 0x0000ff00) << 8) | ((num_items & 0x000000ff) << 24);

  std::vector<int> labels(num_items);
  for (int i = 0; i < num_items; ++i) {
    unsigned char label = 0;
    file.read((char *)&label, sizeof(label));
    labels[i] = static_cast<int>(label);
  }

  return labels;
}

// 读取模型参数
std::vector<float> read_param(const std::string &path) {
  std::ifstream file(path);
  std::vector<float> params;
  float param;
  while (file >> param) {
    params.push_back(param);
  }
  return params;
}

// // 范例kernel函数，无实际作用
// __global__ void add_arrays(int *a, int *b, int *c, int size) {
//   int index = threadIdx.x + blockIdx.x * blockDim.x;
//   if (index < size) {
//     c[index] = a[index] + b[index];
//   }
// }

#define ID4(i0, i1, i2, i3, d0, d1, d2, d3) (((i0) * (d1) + (i1)) * (d2) + (i2)) * (d3) + (i3)

#define ID3(i0, i1, i2, d0, d1, d2) ((i0) * (d1) + (i1)) * (d2) + (i2)

// ReLU激活函数
inline __device__ __host__ float relu(float x) { return x > 0 ? x : 0; }

// 定义卷积层的前向传播函数
void batch_conv2d_f(const float *input, const int batch_count, const int in_channels,
                    const int in_height, const int in_width, const float *weight,
                    const int out_channels, const int kernel_size, const float *bias,
                    const int stride, const int padding, float *output) {

  int out_height = (in_height - kernel_size + 2 * padding) / stride + 1;
  int out_width = (in_width - kernel_size + 2 * padding) / stride + 1;

  // 输入为 (N,C,H,W)

  // 对输入图像进行卷积操作
  // 一个 block 处理一个 (C,H,W)
  for (int b = 0; b < batch_count; b++) {

    for (int k = 0; k < out_channels; k++) {
      for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {

          float sum = 0.0;
          for (int c = 0; c < in_channels; c++) {

            for (int m = 0; m < kernel_size; m++) {
              for (int n = 0; n < kernel_size; n++) {
                int input_row = i * stride + m - padding;
                int input_col = j * stride + n - padding;

                if (input_row >= 0 && input_row < in_height && input_col >= 0 &&
                    input_col < in_width) {
                  sum +=
                      (input[ID4(b, c, input_row, input_col, batch_count, in_channels, in_height,
                                 in_width)] -
                       0.5) /
                      0.5 *
                      weight[ID4(k, c, m, n, out_channels, in_channels, kernel_size, kernel_size)];
                }
              }
            }
          }
          output[ID4(b, k, i, j, batch_count, out_channels, out_height, out_width)] =
              relu(sum + bias[k]);
        }
      }
    }
  }
}

template <int KerSize, int stride, int padding, int in_channels, int in_height, int in_width,
          int out_channels>
__global__ void batch_conv2d_f_ker(float *input, const int batch_count, const int _in_channels,
                                   const int _in_height, const int _in_width, const float *weight,
                                   const int _out_channels, const int kernel_size,
                                   const float *bias, const int _stride, const int _padding,
                                   float *output, const int en_norm) {

  const int out_height = (in_height - KerSize + 2 * padding) / stride + 1;
  const int out_width = (in_width - KerSize + 2 * padding) / stride + 1;

  const int b = blockIdx.x * 4;
  const int h = threadIdx.x / out_width;
  const int w = threadIdx.x % out_width;

  // __shared__ float input_share[in_channels * in_height * in_width];
  // for (int i = threadIdx.x; i < in_channels * in_height * in_width; i += blockDim.x) {
  //   input_share[i] = input[b * in_channels * in_height * in_width + i];
  // }
  // __syncthreads();

  static_assert(out_channels == 1);
  static_assert(in_channels == 1);
  static_assert(padding == 0);

  float weight_reg[KerSize][KerSize];
  for (int m = 0; m < KerSize; m++) {
    for (int n = 0; n < KerSize; n++) {
      weight_reg[m][n] = weight[ID4(0, 0, m, n, out_channels, in_channels, KerSize, KerSize)];
    }
  }

  const int b_tile = 4;

  for (int k = 0; k < out_channels; k++) {

    // for (int h = 0; i < out_height; i++) {
    //   for (int j = 0; j < out_width; j++) {

    float sum[b_tile] = {0.0};
    for (int c = 0; c < in_channels; c++) {

#pragma unroll
      for (int bb = 0; bb < b_tile; bb++)
#pragma unroll
        for (int m = 0; m < KerSize; m++) {
#pragma unroll
          for (int n = 0; n < KerSize; n++) {
            // int n = 0;
            int input_row = h * stride + m - padding;
            int input_col = w * stride + n - padding;

            sum[bb] += (input[(b + bb) * in_channels * in_height * in_width +
                              input_row * in_weight + input_col]) *
                       weight_reg[m][n];
          }
        }
    }
    for (int bb = 0; bb < b_tile; bb++)
      output[ID4(b + bb, k, h, w, batch_count, out_channels, out_height, out_width)] =
          relu(sum[bb] + bias[k]);
    //   }
    // }
  }
  // }
}
// 线性层（Linear）
template <int InSize, int OutSize>
void batch_linear(float *input, float *weight, float *bias, float *output, const int batch_count,
                  const int en_relu) {
  for (int b = 0; b < batch_count; b++) {
    for (int i = 0; i < OutSize; i++) {
      float sum = 0.0;
      for (int j = 0; j < InSize; j++) {
        sum += input[InSize * b + j] * weight[i * InSize + j];
      }
      output[OutSize * b + i] = sum + bias[i];
      if (en_relu)
        output[OutSize * b + i] = relu(output[OutSize * b + i]);
    }
  }
}

void gemmNT_T(const float *A, const float *B, float *C, const float *bias, const int M, const int N,
              const int K, const int en_relu) {
  for (int m = 0; m < M; m++) {   // OutSize
    for (int n = 0; n < N; n++) { // batch_count
      float sum = 0.0;
      for (int k = 0; k < K; k++) { // InSize
        sum += A[m * K + k] * B[n * K + k];
      }
      C[n * M + m] = sum + bias[m];
      if (en_relu)
        C[n * M + m] = relu(C[n * M + m]);
    }
  }
}

template <int BLOCKX, int BLOCKY, int m_tile, int n_tile>
__global__ void gemmNT_T_ker(const float *A, const float *B, float *C, const float *bias,
                             const int M, const int N, const int K, const int en_relu) {
  const int ml = blockIdx.x * BLOCKX + threadIdx.x * m_tile;
  const int nl = blockIdx.y * BLOCKY + threadIdx.y * n_tile;

  // const int m = threadIdx.x;
  // const int n = blockIdx.x;

  const int BLOCK_K = 32;
  __shared__ float As[BLOCKX][BLOCK_K + 1];
  __shared__ float Bs[BLOCKY][BLOCK_K + 1];

  // for (int m = 0; m < M; m++) {   // OutSize
  //   for (int n = 0; n < N; n++) { // batch_count

  const int lid = threadIdx.y * (BLOCKX / m_tile) + threadIdx.x;
  const int lsize = (BLOCKX / m_tile) * (BLOCKY / n_tile);

  int kmax = (K / BLOCK_K) * BLOCK_K;

  float sum[m_tile][n_tile] = {0.0};
  for (int bkIdx = 0; bkIdx < kmax; bkIdx += BLOCK_K) {
    __syncthreads();
    for (int i = lid; i < BLOCKX * BLOCK_K; i += lsize) {
      int m = blockIdx.x * BLOCKX + i / BLOCK_K;
      int k = bkIdx + i % BLOCK_K;
      // if(m < M)
      As[i / BLOCK_K][i % BLOCK_K] = A[m * K + k];
      // else
      //   As[i / BLOCK_K][i % BLOCK_K] = 0;
    }
    for (int i = lid; i < BLOCKY * BLOCK_K; i += lsize) {
      int n = blockIdx.y * BLOCKY + i / BLOCK_K;
      int k = bkIdx + i % BLOCK_K;
      if (n < N)
        Bs[i / BLOCK_K][i % BLOCK_K] = B[n * K + k];
      else
        Bs[i / BLOCK_K][i % BLOCK_K] = 0;
    }
    __syncthreads();

    for (int k = 0; k < BLOCK_K; k++) { // InSize
      // float A_reg[2];
      // #pragma unroll
      // for(int mm = 0; mm<m_tile; mm++)
      //   A_reg[mm] = As[threadIdx.x * m_tile + mm][k];

      // #pragma unroll
      for (int nn = 0; nn < n_tile; nn++) {
        float B_reg = Bs[threadIdx.y * n_tile + nn][k];
        // #pragma unroll
        for (int mm = 0; mm < m_tile; mm++) {
          sum[mm][nn] += As[threadIdx.x * m_tile + mm][k] * B_reg;
          // sum[mm][nn] += A_reg[mm] * B_reg;
        }
      }
    }
  }
  // C[n * M + m] = sum;

  for (int nn = 0; nn < n_tile; nn++) {
    int n = nl + nn;
    for (int mm = 0; mm < m_tile; mm++) {
      int m = ml + mm;
      if (m < M && n < N) {
        // float sum = 0.0;
        for (int k = kmax; k < K; k++) { // InSize
          sum[mm][nn] += A[m * K + k] * B[n * K + k];
        }
        C[n * M + m] = sum[mm][nn] + bias[m];
        if (en_relu)
          C[n * M + m] = relu(C[n * M + m]);
      }
    }
  }
}

#define WIDTH 28
#define HEIGHT 28

void normalize(float *data, int size, float mean, float std) {
  for (int i = 0; i < size; i++) {
    data[i] = (data[i] - mean) / std;
  }
}

__global__ void normalize_ker(float *data, int size, float mean, float std) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    data[i] = (data[i] - mean) / std;
}

template <int ITEMS_PRE_THREAD, int BLOCK_SIZE>
__global__ void vec_max_index_ker(float *data, float *out_data, const int size) {
  const int common_offset = (blockIdx.x * blockDim.x) * ITEMS_PRE_THREAD;
  float max_item = data[common_offset + ITEMS_PRE_THREAD * threadIdx.x];
  int id = 0;
  __shared__ float data_shr[ITEMS_PRE_THREAD * BLOCK_SIZE];
  for(int i=0; i<ITEMS_PRE_THREAD; i++){
    data_shr[threadIdx.x + i * BLOCK_SIZE] = data[common_offset + threadIdx.x + i * BLOCK_SIZE];
  }
  __syncthreads();
  for (int i = 1; i < ITEMS_PRE_THREAD; i++) {
    // 这里可能有分支分歧但不管了，希望编译器能优化掉
    float cmp = data_shr[ITEMS_PRE_THREAD * threadIdx.x + i];
    if (cmp > max_item) {
      max_item = cmp;
      id = i;
    }
  }

  if (blockIdx.x * blockDim.x + threadIdx.x < size)
    out_data[blockIdx.x * blockDim.x + threadIdx.x] = id;
}

void output_tensor(float *data, std::vector<int> shape) {
  int dim = shape.size();

  std::vector<int> off(dim, 0);
  int size = 1;
  for (int i = 0; i < dim; i++)
    size *= shape[i];

  for (int i = 0; i < size; i++) {
    if (off[dim - 1] == 0) {
      for (int i = 0; i < dim; i++)
        printf("%3d, ", off[i]);
      printf("[");
    }
    printf("%+7.4f, ", data[i]);
    off[dim - 1]++;

    if (off[dim - 1] >= shape[dim - 1])
      printf("], \n");

    int ii = dim - 1;
    while (off[ii] >= shape[ii]) {
      off[ii] = 0;
      ii--;
      if (ii < 0)
        break; // end
      off[ii]++;
    }
  }
  printf("\n");
}

#define MIN(x, y) ((x) < (y) ? (x) : (y))

int main(int argc, char *argv[]) {
  std::string dir = argv[1];
  //     第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
  //          // cout << dir;

  // std::string dir = "./";
  // 读取测试集，对于想实现CUDA
  // C/C++训练的同学，参考训练集文件名为train-images-idx3-ubyte和train-labels-idx1-ubyte
  auto images = read_mnist_images(dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte");
  float *images_data = (float *)malloc(sizeof(float) * images.size() * images[0].size());
  for (size_t i = 0; i < images.size(); i++) {
    memcpy(images_data + i * images[0].size(), images[i].data(), images[0].size() * sizeof(float));
  }

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
  // auto fc3_weight = read_param(dir + "/fc3.weight.txt");
  // auto fc3_bias = read_param(dir + "/fc3.bias.txt");

  float *conv1_weight_dev;
  float *conv1_bias_dev;
  float *conv2_weight_dev;
  float *conv2_bias_dev;
  float *fc1_weight_dev;
  float *fc1_bias_dev;
  float *fc2_weight_dev;
  float *fc2_bias_dev;

  checkCudaErrors(cudaMalloc((void **)&conv1_weight_dev, sizeof(float) * conv1_weight.size()));
  checkCudaErrors(cudaMalloc((void **)&conv1_bias_dev, sizeof(float) * conv1_bias.size()));
  checkCudaErrors(cudaMalloc((void **)&conv2_weight_dev, sizeof(float) * conv2_weight.size()));
  checkCudaErrors(cudaMalloc((void **)&conv2_bias_dev, sizeof(float) * conv2_bias.size()));
  checkCudaErrors(cudaMalloc((void **)&fc1_weight_dev, sizeof(float) * fc1_weight.size()));
  checkCudaErrors(cudaMalloc((void **)&fc1_bias_dev, sizeof(float) * fc1_bias.size()));
  checkCudaErrors(cudaMalloc((void **)&fc2_weight_dev, sizeof(float) * fc2_weight.size()));
  checkCudaErrors(cudaMalloc((void **)&fc2_bias_dev, sizeof(float) * fc2_bias.size()));

  checkCudaErrors(cudaMemcpy(conv1_weight_dev, conv1_weight.data(),
                             sizeof(float) * conv1_weight.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(conv1_bias_dev, conv1_bias.data(), sizeof(float) * conv1_bias.size(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(conv2_weight_dev, conv2_weight.data(),
                             sizeof(float) * conv2_weight.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(conv2_bias_dev, conv2_bias.data(), sizeof(float) * conv2_bias.size(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fc1_weight_dev, fc1_weight.data(), sizeof(float) * fc1_weight.size(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fc1_bias_dev, fc1_bias.data(), sizeof(float) * fc1_bias.size(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fc2_weight_dev, fc2_weight.data(), sizeof(float) * fc2_weight.size(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fc2_bias_dev, fc2_bias.data(), sizeof(float) * fc2_bias.size(),
                             cudaMemcpyHostToDevice));

  int correct = 0;
  std::vector<float> pred_labels(labels.size(), -1);

  const int conv_1_out_size = (conv_1_out_channel * conv_1_out_height * conv_1_out_width);

  float *conv_1_out = (float *)malloc(sizeof(float) * images.size() * conv_1_out_size);
  float *linear_1_out = (float *)malloc(sizeof(float) * images.size() * linear_1_out_size);
  float *linear_2_out =
      (float *)malloc(sizeof(float) * images.size() * linear_2_out_size); // 完整大小

  float *conv_1_out_dev;
  float *linear_1_out_dev;
  float *linear_2_out_dev;

  checkCudaErrors(
      cudaMalloc((void **)&conv_1_out_dev, sizeof(float) * batch_size * conv_1_out_size));
  checkCudaErrors(
      cudaMalloc((void **)&linear_1_out_dev, sizeof(float) * images.size() * linear_1_out_size));
  checkCudaErrors(cudaMalloc((void **)&linear_2_out_dev,
                             sizeof(float) * images.size() * linear_2_out_size)); // 完整大小

  float *images_data_dev_ext;
  checkCudaErrors(cudaMalloc((void **)&images_data_dev_ext,
                             sizeof(float) * (images.size() + 2) * (images[0].size() + 2)));
  float *images_data_dev = images_data_dev_ext + (images[0].size() + 2) * 2 + 2;
  checkCudaErrors(cudaMemcpy(images_data_dev, images_data,
                             sizeof(float) * images.size() * images[0].size(),
                             cudaMemcpyHostToDevice));

  // 开始计时，使用chrono计时，不支持其它计时方式
  auto start = std::chrono::high_resolution_clock::now();
  timeprof_start_("all");

  for (size_t batch_off = 0; batch_off < images.size(); batch_off += batch_size) {
    int batch_count = MIN(batch_off + batch_size, images.size()) - batch_off;
#ifdef PROF_CUDA_EVENT
    float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); // 创建Event
    cudaEventCreate(&stop);
#endif

    timeprof_start_("batch_conv2d_f_1");
#ifdef PROF_CUDA_EVENT
    cudaEventRecord(start, 0);
#endif
    {
      batch_conv2d_f_ker<2, conv_1_stride, conv_1_padding, in_channel, in_height, in_weight,
                         conv_1_out_channel>
          <<<batch_count / 4, conv_1_out_height * conv_1_out_width>>>(
              images_data_dev + images[0].size() * batch_off, batch_count, in_channel, in_height,
              in_weight, conv1_weight_dev, conv_1_out_channel, conv_1_kernel_size, conv1_bias_dev,
              conv_1_stride, conv_1_padding, conv_1_out_dev, 1);
    }
#ifdef PROF_CUDA_EVENT
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start); // Waits for an event to complete.
    cudaEventSynchronize(stop);  // Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
    printf("batch_conv2d_f_ker 执行时间：%f(ms)\n", time_elapsed);
#endif
    checkCudaErrors(cudaGetLastError());
    timeprof_end_();
    const int batch_tile = 8;
    const int m_tile = 2;

    timeprof_start_("batch_linear_1");
#ifdef PROF_CUDA_EVENT
    cudaEventRecord(start, 0);
#endif
    {
      dim3 gridDim2(CEIL_DIV(linear_1_out_size, 16 * m_tile), CEIL_DIV(batch_count, 16 * batch_tile));
      dim3 blockDim2(16, 16);
      gemmNT_T_ker<16 * m_tile, 16 * batch_tile, m_tile, batch_tile>
          <<<gridDim2, blockDim2>>>(fc1_weight_dev, conv_1_out_dev, linear_1_out_dev, fc1_bias_dev,
                                    linear_1_out_size, batch_count, linear_1_in_size, 1);
    }
#ifdef PROF_CUDA_EVENT
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start); // Waits for an event to complete.
    cudaEventSynchronize(stop);  // Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
    printf("执行时间：%f(ms)\n", time_elapsed);
#endif
    // printf("\n");
    checkCudaErrors(cudaGetLastError());
    timeprof_end_();

    timeprof_start_("batch_linear_2");
#ifdef PROF_CUDA_EVENT
    cudaEventRecord(start, 0);
#endif
    {
      const int batch_tile = 1;
      dim3 gridDim2(CEIL_DIV(linear_2_out_size, 10), CEIL_DIV(batch_count, 16 * batch_tile));
      dim3 blockDim2(10, 16);
      gemmNT_T_ker<10, 16 * batch_tile, 1, batch_tile><<<gridDim2, blockDim2>>>(
          fc2_weight_dev, linear_1_out_dev, linear_2_out_dev + batch_off * linear_2_out_size,
          fc2_bias_dev, linear_2_out_size, batch_count, linear_1_out_size, 0);
    }
#ifdef PROF_CUDA_EVENT
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start); // Waits for an event to complete.
    cudaEventSynchronize(stop);  // Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
    printf("执行时间：%f(ms)\n", time_elapsed);
#endif
    // printf("\n");
    checkCudaErrors(cudaGetLastError());
    timeprof_end_();

    timeprof_start_("vec_max_index_ker");
#ifdef PROF_CUDA_EVENT
    cudaEventRecord(start, 0);
#endif
    vec_max_index_ker<10, 128>
        <<<CEIL_DIV(batch_count, 128), 128>>>(linear_2_out_dev + batch_off * linear_2_out_size,
                                              linear_2_out_dev + batch_off, batch_count);
    checkCudaErrors(cudaGetLastError());
#ifdef PROF_CUDA_EVENT
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start); // Waits for an event to complete.
    cudaEventSynchronize(stop);  // Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed, start, stop); // 计算时间差
    printf("执行时间：%f(ms)\n", time_elapsed);
#endif
    // printf("\n");
    checkCudaErrors(cudaGetLastError());
    timeprof_end_();
  }
  timeprof_start_("cudaMemcpy linear_2_out");
  checkCudaErrors(cudaMemcpy(linear_2_out, linear_2_out_dev, sizeof(float) * images.size(),
                             cudaMemcpyDeviceToHost));
  timeprof_end_();

  timeprof_start_("pred_labels");

  for (size_t batch_off = 0; batch_off < images.size(); batch_off += batch_size) {
    for (size_t t = batch_off; t < MIN(batch_off + batch_size, images.size()); t++) {
      int pred_label = linear_2_out[t];
      // printf("pred_label=%d\n", pred_label);
      if (pred_label == labels[t]) {
        correct++;
      }
    }
  }
  timeprof_end_();

  timeprof_start_("free");
  // free(conv_1_out);
  // free(maxpool2d_1_out);
  // free(conv_2_out);
  // free(maxpool2d_2_out);
  // free(linear_1_out);
  // free(linear_2_out);

  // checkCudaErrors(cudaFree(conv_1_out_dev));
  // checkCudaErrors(cudaFree(maxpool2d_1_out_dev));
  // checkCudaErrors(cudaFree(conv_2_out_dev));
  // checkCudaErrors(cudaFree(maxpool2d_2_out_dev));
  // checkCudaErrors(cudaFree(linear_1_out_dev));
  // checkCudaErrors(cudaFree(linear_2_out_dev));

  // printf("correct: %d\n", correct);

  timeprof_end_();

  // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
  cudaDeviceSynchronize();

  // 结束计时
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  timeprof_end_();

  // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
  std::cout << std::fixed << std::setprecision(6) << diff.count() << ":" << std::setprecision(4)
            << (double)correct / labels.size() << std::endl;

  timeprof_print_frame_sorted_();

  return 0;
}