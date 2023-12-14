// Load weight from constant memory in FC.
void __global__ FCProcessing2(const float *input, float *output)
{
    const int t = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < FC1; i++)
    {
        sum += input[i] * FC2_WEIGHT[t][i];
    }
    sum += FC2_BIAS[t];
    if (sum < 0.0f)
        sum = 0.0f;
    output[t] = sum;
}

void __global__ FCProcessing3(const float *input, int *prediction)
{
    __shared__ float output_s[FC3];
    const int t = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < FC2; i++)
        sum += input[i] * FC3_WEIGHT[t][i];
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
        *prediction = bestId;
    }
}


// Use shared memory in FC.

void __global__ FCProcessing1(const float *input, const float *weight, float *output)
{
    __shared__ float input_s[FC0];
    const int t = threadIdx.x;
    if (t < FC0) 
        input_s[t] = input[t];
    __syncthreads();

    if (t < FC1)
    {
        float sum = 0.0f;
        for (int i = 0; i < FC0; i++)
        {
            sum += input_s[i] * weight[t * FC0 + i];
        }
        sum += FC1_BIAS[t];
        if (sum < 0.0f)
            sum = 0.0f;
        output[t] = sum;
    }
}

void __global__ FCProcessing2(const float *input, const float *weight, float *output)
{
    __shared__ float input_s[FC1];
    const int t = threadIdx.x;
    if (t < FC1) 
        input_s[t] = input[t];
    __syncthreads();

    if (t < FC2)
    {
        float sum = 0.0f;
        for (int i = 0; i < FC1; i++)
        {
            sum += input_s[i] * weight[t * FC1 + i];
        }
        sum += FC2_BIAS[t];
        if (sum < 0.0f)
            sum = 0.0f;
        output[t] = sum;
    }
}

void __global__ FCProcessing3(const float *input, const float *weight, int *prediction)
{
    __shared__ float input_s[FC2];
    __shared__ float output_s[FC3];
    const int t = threadIdx.x;
    if (t < FC2) 
        input_s[t] = input[t];
    __syncthreads();

    if (t < FC3)
    {
        float sum = 0.0f;
        for (int i = 0; i < FC2; i++)
            sum += input_s[i] * weight[t * FC2 + i];
        output_s[t] = sum;
    }
    __syncthreads();  
    // printf("debug %d %.6f\n", t, sum);
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
        *prediction = bestId;
    }
}