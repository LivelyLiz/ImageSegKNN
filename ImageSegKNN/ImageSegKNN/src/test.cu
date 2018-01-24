#pragma once
#include "header/cuda_util.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "header/cuda_KNN.cuh"
#include "header/cuda_MinHeap.cuh"
#include "header/ppma_io.hpp"
#include <algorithm>
#include <chrono>

// to compile a header into this main, you have to set 
//Configuration Properties -> CUDA C/C++ -> Common -> Generate Relocatable Device Code -> Yes (-rdc=true)

// setup KNN instances for specific images
__host__ KNN<3>* MirrorTestKNN()
{
	float* labelColors = (float*)malloc(sizeof(float) * 3 * 3);
	float* red = RgbLab::MakeColor(255, 0, 0);
	float* green = RgbLab::MakeColor(0, 255, 0);
	float* blue = RgbLab::MakeColor(0, 0, 255);

	for (int i = 0; i < 3; ++i)
	{
		labelColors[0 * 3 + i] = red[i];
		labelColors[1 * 3 + i] = green[i];
		labelColors[2 * 3 + i] = blue[i];
	}

	int numColorsPerLabel = 8;

	KNN<3>* knn = new KNN<3>(labelColors);

	float** label1colors = (float**)malloc(sizeof(float*) * numColorsPerLabel);
	float** label2colors = (float**)malloc(sizeof(float*) * numColorsPerLabel);
	float** label3colors = (float**)malloc(sizeof(float*) * numColorsPerLabel);

	label2colors[0] = RgbLab::MakeColor(255, 255, 255);
	label2colors[1] = RgbLab::MakeColor(250, 250, 250);
	label2colors[2] = RgbLab::MakeColor(162, 163, 168);
	label2colors[3] = RgbLab::MakeColor(216, 211, 207);

	label3colors[0] = RgbLab::MakeColor(83, 145, 255);
	label3colors[1] = RgbLab::MakeColor(104, 182, 255);
	label3colors[2] = RgbLab::MakeColor(173, 125, 89);
	label3colors[3] = RgbLab::MakeColor(248, 194, 150);

	label1colors[0] = RgbLab::MakeColor(254, 254, 156);
	label1colors[1] = RgbLab::MakeColor(226, 77, 21);
	label1colors[2] = RgbLab::MakeColor(255, 164, 43);
	label1colors[3] = RgbLab::MakeColor(159, 65, 4);

	knn->AddColorsToTrainingsset(label1colors, 0, 4);
	knn->AddColorsToTrainingsset(label2colors, 1, 4);
	knn->AddColorsToTrainingsset(label3colors, 2, 4);

	return knn;
}

__host__ KNN<3>* TreeTestKNN()
{
	float* labelColors = (float*)malloc(sizeof(float) * 3 * 3);
	float* black = RgbLab::MakeColor(0, 0, 0);
	float* green = RgbLab::MakeColor(0, 255, 0);
	float* blue = RgbLab::MakeColor(0, 0, 255);

	for (int i = 0; i < 3; ++i)
	{
		labelColors[0 * 3 + i] = black[i];
		labelColors[1 * 3 + i] = green[i];
		labelColors[2 * 3 + i] = blue[i];
	}

	int numColorsPerLabel = 10;

	KNN<3>* knn = new KNN<3>(labelColors);

	float** label1colors = (float**)malloc(sizeof(float*) * numColorsPerLabel);
	float** label2colors = (float**)malloc(sizeof(float*) * numColorsPerLabel);
	float** label3colors = (float**)malloc(sizeof(float*) * numColorsPerLabel);

	for(int i = 0; i < numColorsPerLabel; ++i)
	{
		float lerp = ((float) i / (float) numColorsPerLabel);
		label1colors[i] = RgbLab::MakeColor((int)(lerp * 50.0), (int)(lerp * 50.0), (int)(lerp * 20.0));
		label2colors[i] = RgbLab::MakeColor((1-lerp) * 255.0f + lerp * 60.0f, (1 - lerp) * 255.0f + lerp * 100.0f, (1 - lerp) * 20);
		label3colors[i] = RgbLab::MakeColor((1-lerp) * 255.0f + lerp * 160.0f, (1 - lerp) * 255.0f + lerp * 190, (1 - lerp) * 255.0f + lerp * 255);
	}

	knn->AddColorsToTrainingsset(label1colors, 0, numColorsPerLabel);
	knn->AddColorsToTrainingsset(label2colors, 1, numColorsPerLabel);
	knn->AddColorsToTrainingsset(label3colors, 2, numColorsPerLabel);

	return knn;
}

//helper function
__device__ int getIndexInTrainingsSet(int label, int index, int numColorsPerLabel)
{
	return label * numColorsPerLabel * 3 + index * 3;
}


//****************************************naive kernel***********************************
__global__ void naiveKNNRgb(float* picturedata, int numPixels, int k, int labelCount, 
	float* labelColors, float* trainingsSet, int* trainingsEntryCount, int numColorsPerlabel, 
		int numTrainingsEntries, NeighbourEntry* neighbour_entry, float* voteCount) {

	int gtid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gtid >= numPixels)
	{
		return;
	}

	for(int i = 0; i < k; ++i)
	{
		neighbour_entry[gtid * k + i] = NeighbourEntry(10000, 0);
	}

	//avoid false results by using at max the amount of trainingsentries
	int newk = k;
	if (numTrainingsEntries < newk)
	{
		newk = numTrainingsEntries;
	}

	for(int i = 0; i < labelCount; ++i)
	{
		voteCount[gtid * labelCount + i] = 0;
	}
	voteCount[gtid * labelCount] = newk * 1.0f / 10000.0f ;

	for(int i = 0; i < labelCount; ++i)
	{
		for (int j = 0; j < trainingsEntryCount[i]; ++j)
		{
			// get the distance
			float length = RgbLab::ColorDistance(&picturedata[gtid*3], &trainingsSet[getIndexInTrainingsSet(i, j, numColorsPerlabel)]);
			
			for (int l = 0; l < newk; ++l)
			{
				// if we find something closer than the latest k nearest
				// update our list
				if (length < neighbour_entry[gtid * k + l].distance)
				{
					// update the votes
					voteCount[gtid * labelCount + i] += 1.0 / (length + 0.0000001);
					voteCount[gtid * labelCount + neighbour_entry[gtid * k + newk - 1].label] -= 1.0 / neighbour_entry[gtid * k + newk - 1].distance;

					// therefore we have to insert the new and push the ones behind it one index further (aka copy them)
					for (int m = newk - 2; m > l; --m)
					{
						neighbour_entry[gtid * k + m + 1] = neighbour_entry[gtid * k + m];
					}

					neighbour_entry[gtid * k + l] = NeighbourEntry((length + 0.0000001), i);

					break;
				}
			}
		}
	}

	// determine the right label based on votes
	float maxVote = 0;
	int winningLabel = 0;

	for (int i = 0; i < labelCount; ++i)
	{
		if (maxVote < voteCount[gtid * labelCount + i])
		{
			maxVote = voteCount[gtid * labelCount + i];
			winningLabel = i;
		}
	}

	for(int i = 0; i < 3; ++i)
	{
		picturedata[gtid * 3 + i] = labelColors[winningLabel * 3 + i];
	}
}


//************************************template parameter kernel*******************************
template<int k, int labelCount>
__global__ void templateparamsKNNRgb(float* picturedata, int numPixels,
	float* labelColors, float* trainingsSet, int* trainingsEntryCount, int numColorsPerlabel,
	int numTrainingsEntries) {

	int gtid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gtid >= numPixels)
	{
		return;
	}
	
	NeighbourEntry neighbour_entry[k];
	float voteCount[labelCount];

	for (int i = 0; i < k; ++i)
	{
		neighbour_entry[i] = NeighbourEntry(10000, 0);
	}

	//avoid false results by using at max the amount of trainingsentries
	int newk = k;
	if (numTrainingsEntries < newk)
	{
		newk = numTrainingsEntries;
	}

	for (int i = 0; i < labelCount; ++i)
	{
		voteCount[i] = 0;
	}
	voteCount[0] = 1.0f / 10000.0f * newk;

	for (int i = 0; i < labelCount; ++i)
	{
		for (int j = 0; j < trainingsEntryCount[i]; ++j)
		{
			// get the distance
			float length = RgbLab::ColorDistance(&picturedata[gtid * 3], &trainingsSet[getIndexInTrainingsSet(i, j, numColorsPerlabel)]);

			for (int l = 0; l < newk; ++l)
			{
				// if we find something closer than the latest k nearest
				// update our list
				if (length < neighbour_entry[l].distance)
				{
					// update the votes
					voteCount[i] += 1.0 / (length + 0.0000001);
					voteCount[neighbour_entry[newk - 1].label] -= 1.0 / neighbour_entry[newk - 1].distance;
					
					// therefore we have to insert the new and push the ones behind it one index further (aka copy them)
					for (int m = newk - 2; m > l; --m)
					{
						neighbour_entry[m + 1] = neighbour_entry[m];
					}
					neighbour_entry[l] = NeighbourEntry((length + 0.0000001), i);
					break;
				}
			}
		}
	}

	// determine the right label based on votes
	float maxVote = 0;
	int winningLabel = 0;

	for (int i = 0; i < labelCount; ++i)
	{
		if (maxVote < voteCount[i])
		{
			maxVote = voteCount[i];
			winningLabel = i;
		}
	}

	for (int i = 0; i < 3; ++i)
	{
		picturedata[gtid * 3 + i] = labelColors[winningLabel * 3 + i];
	}
}

//*******************************shared memory template parameter*****************************
template<int k, int labelCount>
__global__ void sharedKNNRgb(float* picturedata, int numPixels,
	float* labelColors, float* trainingsSet, int* trainingsEntryCount, int numColorsPerlabel,
	int numTrainingsEntries) {

	__shared__ float sTrainingsSet[200*3];
	__shared__ int sTrainingsEntryCount[labelCount];
	__shared__ float sLabelColors[labelCount * 3];

	int gtid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gtid >= numPixels)
	{
		return;
	}

	if(threadIdx.x < labelCount)
	{
		//printf("thread: %i, color %f, %f, %f \n", threadIdx.x, labelColors[threadIdx.x * 3], labelColors[threadIdx.x * 3 + 1], labelColors[threadIdx.x * 3 + 2]);
		sLabelColors[threadIdx.x * 3] = labelColors[threadIdx.x * 3];
		sLabelColors[threadIdx.x * 3 + 1] = labelColors[threadIdx.x * 3 + 1];
		sLabelColors[threadIdx.x * 3 + 2] = labelColors[threadIdx.x * 3 + 2];

		//printf("thread: %i, shared color %f, %f, %f \n", threadIdx.x, sLabelColors[threadIdx.x * 3], sLabelColors[threadIdx.x * 3 + 1], sLabelColors[threadIdx.x * 3 + 2]);

		sTrainingsEntryCount[threadIdx.x] = trainingsEntryCount[threadIdx.x];

		for(int i = 0; i < sTrainingsEntryCount[threadIdx.x]; ++i)
		{
			sTrainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel)] = trainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel)];
			sTrainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel) + 1] = trainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel) + 1];
			sTrainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel) + 2] = trainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel) + 2];
		}
	}

	__syncthreads();

	NeighbourEntry neighbour_entry[k];
	float voteCount[labelCount];

	for (int i = 0; i < k; ++i)
	{
		neighbour_entry[i] = NeighbourEntry(10000, 0);
	}

	//avoid false results by using at max the amount of trainingsentries
	int newk = k;
	if (numTrainingsEntries < newk)
	{
		newk = numTrainingsEntries;
	}

	for (int i = 0; i < labelCount; ++i)
	{
		voteCount[i] = 0;
	}
	voteCount[0] = newk * 1.0f / 10000.0f;

	for (int i = 0; i < labelCount; ++i)
	{
		for (int j = 0; j < sTrainingsEntryCount[i]; ++j)
		{
			// get the distance
			float length = RgbLab::ColorDistance(&picturedata[gtid * 3], &sTrainingsSet[getIndexInTrainingsSet(i, j, numColorsPerlabel)]);

			for (int l = 0; l < newk; ++l)
			{
				// if we find something closer than the latest k nearest
				// update our list
				if (length < neighbour_entry[l].distance)
				{
					// update the votes
					voteCount[i] += 1.0 / (length + 0.0000001);
					voteCount[neighbour_entry[newk - 1].label] -= 1.0 / neighbour_entry[newk - 1].distance;

					// therefore we have to insert the new and push the ones behind it one index further (aka copy them)
					for (int m = newk - 2; m > l; --m)
					{
						neighbour_entry[m + 1] = neighbour_entry[m];
					}
					neighbour_entry[l] = NeighbourEntry((length + 0.0000001), i);
					break;
				}
			}
		}
	}

	// determine the right label based on votes
	float maxVote = 0;
	int winningLabel = 0;

	for (int i = 0; i < labelCount; ++i)
	{
		if (maxVote < voteCount[i])
		{
			maxVote = voteCount[i];
			winningLabel = i;
		}
	}

	for (int i = 0; i < 3; ++i)
	{
		//printf("winning label %i , color i=%i %f \n", winningLabel, i, sLabelColors[winningLabel * 3 + i]);
		picturedata[gtid * 3 + i] = sLabelColors[winningLabel * 3 + i];
	}
}

//******************************************split kernel**************************************
__device__ void swap(NeighbourEntry* x, NeighbourEntry* y)
{
	float tempdist = x->distance;
	int templabel = x->label;
	x->distance = y->distance;
	x->label = y->label;
	y->distance = tempdist;
	y->label = templabel;
}

template<int labelCount>
__global__ void computeLength(float* picturedata, NeighbourEntry* neighbours, int numPixels,
	float* trainingsSet, int* trainingsEntryCount, int numColorsPerlabel,
	int numTrainingsEntries) {

	__shared__ float sTrainingsSet[200 * 3];
	__shared__ int sTrainingsEntryCount[labelCount];

	int gtid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gtid >= numPixels)
	{
		return;
	}

	if (threadIdx.x < labelCount)
	{
		sTrainingsEntryCount[threadIdx.x] = trainingsEntryCount[threadIdx.x];

		for (int i = 0; i < sTrainingsEntryCount[threadIdx.x]; ++i)
		{
			sTrainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel)] = trainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel)];
			sTrainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel) + 1] = trainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel) + 1];
			sTrainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel) + 2] = trainingsSet[getIndexInTrainingsSet(threadIdx.x, i, numColorsPerlabel) + 2];
		}
	}

	__syncthreads();

	int index = 0;

	for (int i = 0; i < labelCount; ++i)
	{
		for (int j = 0; j < sTrainingsEntryCount[i]; ++j)
		{
			// get the distance
			neighbours[gtid * numTrainingsEntries + index] = NeighbourEntry(RgbLab::ColorDistance(&picturedata[gtid * 3], &sTrainingsSet[getIndexInTrainingsSet(i, j, numColorsPerlabel)]), i);
			index++;
		}
	}
}



__global__ void bubblesort(NeighbourEntry* neighbours, int numPixels, int stride)
{
	int gtid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gtid >= numPixels)
	{
		return;
	}

	for (int i = 0; i < stride - 1; i++)
	{   
		for (int j = gtid * stride; j < gtid * stride + stride - i - 1; j++)
		{
			if (neighbours[j] > neighbours[j + 1])
			{
				swap(&neighbours[j], &neighbours[j + 1]);
			}
		}	
	}
}

template<int k, int labelCount>
__global__ void writeLabel(float* picturedata, NeighbourEntry* neighbours, int numPixels, float* labelColors, int numTrainingsEntries)
{
	int gtid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gtid >= numPixels)
	{
		return;
	}

	float voteCount[labelCount];

	int newk = k;
	if (numTrainingsEntries < newk)
	{
		newk = numTrainingsEntries;
	}

	for (int i = 0; i < labelCount; ++i)
	{
		voteCount[i] = 0;
	}

	// determine the right label based on votes
	float maxVote = 0;
	int winningLabel = 0;

	for(int i = gtid * numTrainingsEntries; i < gtid * numTrainingsEntries + newk; ++i)
	{
		voteCount[neighbours[i].label] += 1.0f / (neighbours[i].distance + 0.00000001);
	}

	for (int i = 0; i < labelCount; ++i)
	{
		if (maxVote < voteCount[i])
		{
			maxVote = voteCount[i];
			winningLabel = i;
		}
	}

	for (int i = 0; i < 3; ++i)
	{
		picturedata[gtid * 3 + i] = labelColors[winningLabel * 3 + i];
	}
}

//***********************************************************************************
//*********************************  main  ******************************************
//***********************************************************************************
void main()
{
	// get number of CUDA devices
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	std::cerr << "devices found: " << deviceCount << std::endl;

	// set the device
	cudaSetDevice(0);
	
	// query the device properties
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	//specify image
	//try "Tree" or "MirrorsEdgeTest" here
	// -> NEED TO USE THE APPROPRIATE KNN INSTANCE BELOW!!
	std::string imagename = "Tree";
	std::string file = "images/" + imagename + ".ppm";
	int xsize = 0;
	int ysize = 0;
	int maxrgb = 0;

	int* r;
	int* g;
	int* b;

	//make a knn instance to use its data on device
	const int labelcount = 3;

	std::chrono::duration<double, std::milli> allocationcpu = std::chrono::duration<double, std::milli>();
	auto cpustart = std::chrono::high_resolution_clock::now();
	
	//CHANGE KNN INSTANCE HERE (either TreeTestKNN or MirrorTestKNN)
	KNN<labelcount> knn = *TreeTestKNN();
	
	auto cpuend = std::chrono::high_resolution_clock::now();
	allocationcpu = cpuend - cpustart;

	//read in image
	ppma_read(file, xsize, ysize, maxrgb, &r, &g, &b);

	//compute number of pixels from read image
	int numPixels = xsize * ysize;

	//make 3 seperate arrays of r, g, b to one with [r, g, b, r, g, b,...]
	float* host_picturedata = (float*)malloc(sizeof(float) * numPixels * 3);
	for(int i = 0; i < numPixels; ++i)
	{
		host_picturedata[3*i] = r[i] / 255.0f;
		host_picturedata[3*i + 1] = g[i] / 255.0f;
		host_picturedata[3*i + 2] = b[i] / 255.0f;
	}

	//allocate memory for result
	float* res = (float*)malloc(sizeof(float) * numPixels * 3);

	std::chrono::duration<double, std::milli> allocationall = std::chrono::duration<double, std::milli>();
	std::chrono::duration<double, std::milli> allocationsplitted = std::chrono::duration<double, std::milli>();
	std::chrono::duration<double, std::milli> allocationnaive = std::chrono::duration<double, std::milli>();

	//allocate device memory
	auto allocallstart = std::chrono::high_resolution_clock::now();
	float* device_picturedata;
	checkErrorsCuda(cudaMalloc((void**)&device_picturedata, sizeof(float) * numPixels * 3));
	checkErrorsCuda(cudaMemcpy(device_picturedata, host_picturedata, sizeof(float) * numPixels * 3, cudaMemcpyHostToDevice));
	float* device_labelColors;
	checkErrorsCuda(cudaMalloc((void**) &device_labelColors, knn.GetSizeOfLabelColors()));
	checkErrorsCuda(cudaMemcpy(device_labelColors, knn.GetLabelColors(), knn.GetSizeOfLabelColors(), cudaMemcpyHostToDevice));
	float* device_trainingsSet;
	checkErrorsCuda(cudaMalloc((void**) &device_trainingsSet, knn.GetSizeOfTrainingsset()));
	checkErrorsCuda(cudaMemcpy(device_trainingsSet, knn.GetTrainingsSet(), knn.GetSizeOfTrainingsset(), cudaMemcpyHostToDevice));
	int* device_trainingsEntryCount;
	checkErrorsCuda(cudaMalloc((void**)&device_trainingsEntryCount, knn.GetSizeOfTrainingsSetCount()));
	checkErrorsCuda(cudaMemcpy(device_trainingsEntryCount, knn.GetTrainingsSetCount(), knn.GetSizeOfTrainingsSetCount(), cudaMemcpyHostToDevice));
	auto allocallend = std::chrono::high_resolution_clock::now();
	allocationall = allocallend - allocallstart;

	// determine thread layout
	const int max_threads_per_block = props.maxThreadsPerBlock;
	int num_blocks = (numPixels) / max_threads_per_block;
	if ((numPixels) % max_threads_per_block != 0)
	{
		num_blocks++;
	}
	int num_threads_per_block = std::min(numPixels, max_threads_per_block);

	const int k =7;

	//allocate memory to work on in naive kernel
	auto allocnaivestart = std::chrono::high_resolution_clock::now();
	NeighbourEntry* device_neighbourentry;
	checkErrorsCuda(cudaMalloc((void**)&device_neighbourentry, sizeof(NeighbourEntry) * k * numPixels));
	float* device_votecount;
	checkErrorsCuda(cudaMalloc((void**)&device_votecount, sizeof(float) * labelcount * numPixels));
	auto allocnaiveend = std::chrono::high_resolution_clock::now();

	//allocate memory for use in splitted kernels
	NeighbourEntry* device_neighbours;
	checkErrorsCuda(cudaMalloc((void**)&device_neighbours, sizeof(NeighbourEntry) * knn.GetNumTrainingsEntries() * numPixels));	
	auto allocsplittedend = std::chrono::high_resolution_clock::now();
	allocationnaive = allocnaiveend - allocnaivestart;
	allocationsplitted = allocsplittedend - allocnaiveend;

	int* rnew = (int*)malloc(sizeof(int) * xsize * ysize);
	int* gnew = (int*)malloc(sizeof(int) * xsize * ysize);
	int* bnew = (int*)malloc(sizeof(int) * xsize * ysize);

	std::chrono::duration<double, std::milli> timecpu = std::chrono::duration<double, std::milli>();
	std::chrono::duration<double, std::milli> timenaive = std::chrono::duration<double, std::milli>();
	std::chrono::duration<double, std::milli> timetp = std::chrono::duration<double, std::milli>();
	std::chrono::duration<double, std::milli> timeshared = std::chrono::duration<double, std::milli>();
	std::chrono::duration<double, std::milli> timesplitted = std::chrono::duration<double, std::milli>();

	int numWdh = 10;

	for(int wdh = 0; wdh < numWdh; ++wdh)
	{
		std::cout << "iteration " << wdh << std::endl;
		//*************run CPU***********************
		auto timecpustart = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < xsize*ysize; ++i)
		{
			float color[3] = { host_picturedata[i * 3], host_picturedata[i * 3 + 1], host_picturedata[i * 3 + 2] };
			float* labelcolor = knn.GetLabelColor(knn.DetermineLabelRgb<k>(&color[0], true));
			rnew[i] = (int)(labelcolor[0] * 255);
			gnew[i] = (int)(labelcolor[1] * 255);
			bnew[i] = (int)(labelcolor[2] * 255);
		}
		auto timecpuend = std::chrono::high_resolution_clock::now();
		timecpu += timecpuend - timecpustart;
		//*******************************************

		//***********run naive kernel****************
		auto timenaivestart = std::chrono::high_resolution_clock::now();
		naiveKNNRgb << <num_blocks, num_threads_per_block >> > (device_picturedata, numPixels, k, labelcount, device_labelColors, device_trainingsSet, device_trainingsEntryCount, knn.GetNumColorsPerLabel(), knn.GetNumTrainingsEntries(), device_neighbourentry, device_votecount);
		checkErrorsCuda(cudaDeviceSynchronize());
		auto timenaiveend = std::chrono::high_resolution_clock::now();
		timenaive += timenaiveend - timenaivestart;
		//*******************************************

		//*************run template params kernel*******
		checkErrorsCuda(cudaMemcpy(device_picturedata, host_picturedata, sizeof(float) * numPixels * 3, cudaMemcpyHostToDevice));
		auto timetpstart = std::chrono::high_resolution_clock::now();
		templateparamsKNNRgb<k, labelcount> << <num_blocks, num_threads_per_block >> > (device_picturedata, numPixels, device_labelColors, device_trainingsSet, device_trainingsEntryCount, knn.GetNumColorsPerLabel(), knn.GetNumTrainingsEntries());
		checkErrorsCuda(cudaDeviceSynchronize());
		auto timetpend = std::chrono::high_resolution_clock::now();
		timetp += timetpend - timetpstart;
		//**********************************************

		//*************run shared kernel*******
		checkErrorsCuda(cudaMemcpy(device_picturedata, host_picturedata, sizeof(float) * numPixels * 3, cudaMemcpyHostToDevice));
		auto timesharedstart = std::chrono::high_resolution_clock::now();
		sharedKNNRgb<k, labelcount> << <num_blocks, num_threads_per_block >> > (device_picturedata, numPixels, device_labelColors, device_trainingsSet, device_trainingsEntryCount, knn.GetNumColorsPerLabel(), knn.GetNumTrainingsEntries());
		checkErrorsCuda(cudaDeviceSynchronize());
		auto timesharedend = std::chrono::high_resolution_clock::now();
		timeshared += timesharedend - timesharedstart;
		//**********************************************

		//*************run splitted kernel***************
		checkErrorsCuda(cudaMemcpy(device_picturedata, host_picturedata, sizeof(float) * numPixels * 3, cudaMemcpyHostToDevice));
		auto timesplittedstart = std::chrono::high_resolution_clock::now();
		computeLength<labelcount> << <num_blocks, num_threads_per_block >> > (device_picturedata, device_neighbours, numPixels, device_trainingsSet, device_trainingsEntryCount, knn.GetNumColorsPerLabel(), knn.GetNumTrainingsEntries());
		checkErrorsCuda(cudaDeviceSynchronize());
		bubblesort << <num_blocks, num_threads_per_block >> > (device_neighbours, numPixels, knn.GetNumTrainingsEntries());
		checkErrorsCuda(cudaDeviceSynchronize());
		writeLabel<k, labelcount> << <num_blocks, num_threads_per_block >> > (device_picturedata, device_neighbours, numPixels, device_labelColors, knn.GetNumTrainingsEntries());
		checkErrorsCuda(cudaDeviceSynchronize());
		auto timesplittedend = std::chrono::high_resolution_clock::now();
		timesplitted += timesplittedend - timesplittedstart;
		//***********************************************	
	}
	checkLastCudaError("kernel execution failed");

	//********************************time measurement results*******************************
	std::cout << "\nKernel execution time" << std::endl;
	std::cout << "cpu: " << timecpu.count()/numWdh << " ms\nnaive: " << timenaive.count() / numWdh <<
		" ms\ntemplate params: " << timetp.count() / numWdh <<
			" ms\nshared: " << timeshared.count() / numWdh <<
				" ms\nsplitted: "<< timesplitted.count() / numWdh <<
					" ms" << std::endl;

	//get data from device
	checkErrorsCuda(cudaMemcpy(res, device_picturedata, sizeof(float) * numPixels * 3, cudaMemcpyDeviceToHost));

	
	// write data back to 3 intarrays, so it can be written to a file by ppma class
	allocallstart = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < numPixels; ++i)
	{
		r[i] = (int) (res[3*i] * 255);
		g[i] = (int) (res[3*i + 1] * 255);
		b[i] = (int) (res[3*i + 2] * 255);
	}
	allocallend = std::chrono::high_resolution_clock::now();
	allocationall += allocallend - allocallstart;

	std::cout << "\nallocation and setting data layout CPU: " << allocationcpu.count() <<"\nallocation and copying GPU: " << allocationall.count() << " ms\n" <<
		"naive kernel needs additional " << allocationnaive.count() << " ms\n" <<
		"splitted kernels need additional " << allocationsplitted.count() << " ms" << std::endl;

	ppma_write("images/" + imagename + "Seg.ppm", xsize, ysize, r, g, b);
	ppma_write("images/" + imagename + "SegCpu.ppm", xsize, ysize, rnew, gnew, bnew);

	printf("ready\n");

	//free host memory
	free(r);
	free(g);
	free(b);
	free(rnew);
	free(gnew);
	free(bnew);
	free(host_picturedata);

	//free device memory
	cudaFree(device_picturedata);
	cudaFree(device_labelColors);
	cudaFree(device_trainingsEntryCount);
	cudaFree(device_trainingsSet);
	cudaFree(device_neighbourentry);
	cudaFree(device_votecount);
	cudaFree(device_neighbours);
}