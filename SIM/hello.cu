/* file: hello.cu */
#include <stdio.h>
#include <cuda_runtime.h>
__global__ void add(int a, int b, int *c)
{
		for(int i = 0; i < 10000; i++)
		{
	*c = a + b;
	__syncthreads();
		}
}

int main()
{
	    int c;
	        int *dev_c;
		    cudaMalloc((void **)&dev_c, sizeof(int));
				{
		        add<<<1, 1>>>(2, 7, dev_c);
			    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
			     }   printf("2 + 7 = %d\n", c);
					    return 0;
}
