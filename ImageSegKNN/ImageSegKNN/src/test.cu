/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : lecture 8
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: CUDA basics
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <algorithm>

//#include "cuda_util.h"
#include <cuda_runtime.h>
#include <cuda.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Non-local means filter in 1D
////////////////////////////////////////////////////////////////////////////////////////////////////
template< int ApronSize >
void
meanRef(float* data, float* means, int n) {

	for (int i = 0; i < n; ++i) {
		means[i] = 0.0;
		for (int k = i - ApronSize; k <= i + ApronSize; ++k) {
			if (k >= 0 && k < n) {
				means[i] += data[k];
			}
		}
		means[i] /= 2 * ApronSize + 1;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Non-local means filter in 1D
////////////////////////////////////////////////////////////////////////////////////////////////////
template< int ApronSize > //use this as template function, so compiler can optimize the loop
__global__
void
mean(float* gdata, int n) {

	__shared__ float smem[1024 + 2 * ApronSize];

	//have to read from global memory -> use global ID, but write to block specific memory -> use local id
	int gtid = blockDim.x * blockIdx.x + threadIdx.x;
	smem[threadIdx.x + ApronSize] = gdata[gtid];

	//padding
	// very first block
	if (blockIdx.x == 0)
	{
		if (threadIdx.x < ApronSize)
		{
			smem[threadIdx.x] = 0.0f;
		}
	}
	//every other block
	else
	{

	}

	//very last block
	if (blockIdx.x == gridDim.x - 1)
	{
		if (threadIdx.x >= n - ApronSize)
		{
			smem[n + ApronSize + (n - threadIdx.x - 1)] = 0.0f;
		}
	}
	else
	{

	}

	//if more threads than one warp (32) -> need to sync them to avoid errors
	__syncthreads();

	float meanval = 0.0f;
	int start = threadIdx.x - ApronSize;
	int end = threadIdx.x + ApronSize + 1;
	for (int j = start; j < end; j++)
	{
		meanval += smem[j + ApronSize];
	}
	meanval /= static_cast<float> (2 * ApronSize + 1);
	gdata[threadIdx.x] = meanval;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// program entry point
////////////////////////////////////////////////////////////////////////////////////////////////////
int
main(int /*argc*/, char** /*argv*/) {

	const int n = 128;
	float* data = (float*)malloc(sizeof(float) * n);
	for (unsigned int i = 0; i < n; ++i) {
		data[i] = (float)i;
	}

	// compute reference solution
	float* meansref = (float*)malloc(sizeof(float) * n);
	meanRef<2>(data, meansref, n);

	// check execution environment
	// get number of CUDA devices
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount < 1)
	{
		exit(0);
	}

	// query the device properties
	//cudaDeviceProp props;
	//cudaGetDeviceProperties(&props, 0);
	//printDeviceProps(props);

	// set the device
	cudaSetDevice(0);

	// allocate device memory
	float* device_data = nullptr;
	cudaMalloc((void**)&device_data, n * sizeof(float));

	// copy device memory
	cudaMemcpy(device_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

	// determine thread layout
	const int max_threads_per_block = 1024; //should better be read from device properties
	int num_blocks = n / max_threads_per_block;
	if (n % max_threads_per_block != 0)
	{
		num_blocks++;
	}
	int num_threads_per_block = std::min(n, max_threads_per_block);

	// run kernel
	// mean is a template function, so need to put the template argument there
	// third argument would be size if shared memory
	const int apron_size = 2;
	mean<2> << <num_blocks, num_threads_per_block, n + 2 * apron_size >> >(device_data, n);
	checkLastCudaError("kernel execution failed");

	// copy result back to host
	cudaMemcpy(data, device_data, n * sizeof(float), cudaMemcpyDeviceToHost);

	// output
	bool correct = true;
	for (unsigned int i = 0; i < n; ++i) {
		correct &= (data[i] == meansref[i]);
	}
	std::cout << (correct ? "correct" : "incorrect") << std::endl;

	// clean up device memory
	cudaFree(device_data);

	// clean up host memory
	free(meansref);
	free(data);

	std::getchar();

	return EXIT_SUCCESS;
}
