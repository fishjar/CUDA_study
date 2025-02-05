#include <cuda_runtime.h>
#include <stdio.h>
#include "cudastart.h"
int recursiveReduce(int *data, int const size)
{
	// terminate check
	if (size == 1) return data[0];
	// renew the stride
	int const stride = size / 2;
	if (size % 2 == 1)
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
		// 奇数的情况，把最后一个数加到第一个位置
		data[0] += data[size - 1];
	}
	else
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
	}
	// call
	return recursiveReduce(data, stride);
}



__global__ void reduceNeighbored(int * g_idata,int * g_odata,unsigned int n) 
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	//boundary check
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= n) return;
	//convert global data pointer to the 
	int *idata = g_idata + blockIdx.x*blockDim.x;
	//in-place reduction in global memory
	// 这里好像只考虑了 blocksize 为 2^n 的情况
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		// 步长是1,则每2个位置计算一次
		// 步长是2,则每4个位置计算一次
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		// 实际是同步 block 内的 warp, warp 内的 thread 是本来就同步的
		__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}

int main(int argc,char** argv)
{
	initDevice(0);
	
	//initialization

	int size = 1 << 24;
	printf("	with array size %d  ", size);

	//execution configuration
	int blocksize = 1024;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);   //从命令行输入设置block大小
	}
	dim3 block(blocksize, 1);
	dim3 grid((size - 1) / block.x + 1, 1); // 通过blocksize计算gridsize，向上取整
	printf("grid %d block %d \n", grid.x, block.x);

	//allocate host memory
	size_t bytes = size * sizeof(int);
	int *idata_host = (int*)malloc(bytes);
	int *odata_host = (int*)malloc(grid.x * sizeof(int));
	int * tmp = (int*)malloc(bytes);

	//initialize the array
	initialData_int(idata_host, size);

	memcpy(tmp, idata_host, bytes);
	double timeStart, timeElaps;
	int gpu_sum = 0;

	// device memory
	int * idata_dev = NULL;
	int * odata_dev = NULL;
	CHECK(cudaMalloc((void**)&idata_dev, bytes));
	CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int))); // odata_dev的长度是block的数量

	//cpu reduction 对照组
	int cpu_sum = 0;
	timeStart = cpuSecond();
	// 递归算法
	// cpu_sum = recursiveReduce(tmp, size);
	for (int i = 0; i < size; i++)
		cpu_sum += tmp[i];
	timeElaps = 1000*(cpuSecond() - timeStart);

	printf("cpu sum:%d \n", cpu_sum);
	printf("cpu reduction elapsed %lf ms cpu_sum: %d\n", timeElaps, cpu_sum);


	//kernel reduceNeighbored

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	// cudaDeviceSynchronize() ：该方法将停止CPU端线程的执行，直到GPU端完成之前CUDA的任务，
	// 包括kernel函数、数据拷贝等。
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceNeighbored <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceNeighbored elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x, block.x);
    
	// free host memory

	free(idata_host);
	free(odata_host);
	CHECK(cudaFree(idata_dev));
	CHECK(cudaFree(odata_dev));

	//reset device
	cudaDeviceReset();

	//check the results
	if (gpu_sum == cpu_sum)
	{
		printf("Test success!\n");
	}
	return EXIT_SUCCESS;
}
