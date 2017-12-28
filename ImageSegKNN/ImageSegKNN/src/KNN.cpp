#pragma once
#include "header/KNN.h"
#include <cstdlib>
#include "header/cuda_RgbLab.cuh"


KNN::KNN() : labelCount(2) 
{
	//  allocating memory and initializing data

	numColorsPerLabel = 8;
	
	//not sure about this pointer stuff here
	trainingsSet = (float***) std::malloc(sizeof(float**) * labelCount);
	for(int i = 0; i < labelCount; ++i)
	{
		trainingsSet[i] = (float**)malloc(sizeof(float*) * numColorsPerLabel);
	}

	trainingsSetCount = (int*)std::malloc(sizeof(int) * 2);
	trainingsSetCount[0] = 1;
	trainingsSetCount[1] = 1;

	float* white = (float*)malloc(sizeof(float) * 3);
	float* black = (float*)malloc(sizeof(float) * 3);
	float* red = (float*)malloc(sizeof(float) * 3);
	float* blue = (float*)malloc(sizeof(float) * 3);

	for (int i = 0; i < 3; ++i)
	{
		white[i] = 1.0;
		black[i] = 0.0;
		blue[i] = 0.0;
		red[i] = 0.0;
	}

	blue[2] = 1.0;
	red[0] = 1.0;

	trainingsSet[0][0] = &white[0];
    trainingsSet[1][0] = &black[0];

	labelColors = (float**)malloc(sizeof(float) * labelCount * 3);
	labelColors[0] = &blue[0];
	labelColors[1] = &red[0];
}

KNN::KNN(int labelCount, float** labelColors) : labelCount(labelCount), labelColors(labelColors) 
{
	trainingsSetCount = (int*)std::malloc(sizeof(float) * labelCount);

	numColorsPerLabel = 16;

	trainingsSet = (float***) std::malloc(sizeof(float) * labelCount * numColorsPerLabel * 3);
	for (int i = 0; i < labelCount; ++i)
	{
		trainingsSet[i] = (float**)malloc(sizeof(float*) * numColorsPerLabel);
	}
}

KNN::KNN(int labelCount, float** labelColors, float*** ts, int* trainingsSetCount, int maxColorsInLabel) : labelCount(labelCount), labelColors(labelColors), trainingsSet(ts),
	trainingsSetCount(trainingsSetCount), numColorsPerLabel(maxColorsInLabel)
{
	
}

KNN::~KNN()
{
	for(int i = 0; i < labelCount; ++i)
	{
		free(labelColors[i]);
		for(int j = 0; j < trainingsSetCount[i]; ++j)
		{
			free(trainingsSet[i][j]);
		}
		free(trainingsSet[i]);
	}

	free(trainingsSet);
	free(trainingsSetCount);
	free(labelColors);
}

float* KNN::GetLabelColor(int labelID)
{
	if(labelID < labelCount)
	{
		return labelColors[labelID];
	}
	return nullptr;
}

// naiive implementation of finding the k nearest neighbours O(k * n) where n == number of trainingsdata
// any other implementation might be better

int KNN::DetermineLabel(int k, float* data, bool weighted)
{
	float** neighbours = (float**)malloc(sizeof(float) * k * 3);
	float* distances = (float*)malloc(sizeof(float) * k);
	int* labels = (int *)malloc(sizeof(int) * k);
	float* voteCount = (float*)malloc(sizeof(float)*labelCount);

	// initialize allocated memory;
	for (int i = 0; i < labelCount; ++i)
	{
		voteCount[i] = 0;
	}

	for (int i = 0; i < k; ++i)
	{
		distances[i] = 10.0;
		labels[i] = 0;
		neighbours[i] = nullptr;
	}

	// go through each trainingsdata
	for (int i = 0; i < labelCount; ++i)
	{
		for (int j = 0; j < trainingsSetCount[i]; ++j)
		{
			// get the distance
			float length = RgbLab::ColorDistance(data, trainingsSet[i][j], 3);
			for (int l = 0; l < k; ++l)
			{
				// if we find something closer than the latest k nearest
				// update our list
				if (length < distances[l])
				{
					// update the votes
					if (weighted)
					{
						voteCount[i] += 1.0 / length;
						voteCount[labels[k - 1]] -= 1.0 / distances[k - 1];
					}
					else
					{
						voteCount[i] += 1.0;
						voteCount[labels[k - 1]] -= 1.0;
					}

					// therefore we have to insert the new and push the ones behind it one index further (aka copy them)
					for (int m = k - 2; m > l; --m)
					{
						labels[m + 1] = labels[m];
						neighbours[m + 1] = neighbours[m];
						distances[m + 1] = distances[m];
					}
					labels[l] = i;
					neighbours[l] = trainingsSet[i][j];
					distances[l] = length;

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

	free(neighbours);
	free(labels);
	free(distances);
	free(voteCount);

	return winningLabel;
}

// using a non type template function enables compile time optimization
// and it is also possible to allocate memory on the stack instead of heap
// in CUDA this means it is at least possible to have the arrays in the registers
// instead of global memory -> performance improvement!!

// maybe make label count also a template parameter, because vote count will not be in registers otherwise

template <int k>
int KNN::DetermineLabel(float* data, bool weighted)
{
	float* neighbours[k];
	float distances[k];
	int labels[k];
	float* voteCount = (float*)malloc(sizeof(float)*labelCount);;

	// initialize allocated memory;
	for (int i = 0; i < labelCount; ++i)
	{
		voteCount[i] = 0;
	}

	for (int i = 0; i < k; ++i)
	{
		distances[i] = 10.0;
		labels[i] = 0;
		neighbours[i] = nullptr;
	}

	// go through each trainingsdata
	for (int i = 0; i < labelCount; ++i)
	{
		for (int j = 0; j < trainingsSetCount[i]; ++j)
		{
			// get the distance
			float length = RgbLab::ColorDistance(data, trainingsSet[i][j], 3);
			for (int l = 0; l < k; ++l)
			{
				// if we find something closer than the latest k nearest
				// update our list
				if (length < distances[l])
				{
					// update the votes
					if (weighted)
					{
						voteCount[i] += 1.0 / length;
						voteCount[labels[k - 1]] -= 1.0 / distances[k - 1];
					}
					else
					{
						voteCount[i] += 1.0;
						voteCount[labels[k - 1]] -= 1.0;
					}

					// therefore we have to insert the new and push the ones behind it one index further (aka copy them)
					for (int m = k - 2; m > l; --m)
					{
						labels[m + 1] = labels[m];
						neighbours[m + 1] = neighbours[m];
						distances[m + 1] = distances[m];
					}
					labels[l] = i;
					neighbours[l] = trainingsSet[i][j];
					distances[l] = length;

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

	free(voteCount);

	return winningLabel;
}

// needed to be able to link the files
// if any other than those values are used there will be a linker error
// use mainly prime numbers to have the best result
template int KNN::DetermineLabel<1>(float* data, bool weighted);
template int KNN::DetermineLabel<2>(float* data, bool weighted);
template int KNN::DetermineLabel<3>(float* data, bool weighted);
template int KNN::DetermineLabel<5>(float* data, bool weighted);
template int KNN::DetermineLabel<7>(float* data, bool weighted);
template int KNN::DetermineLabel<11>(float* data, bool weighted);
template int KNN::DetermineLabel<13>(float* data, bool weighted);


void KNN::AddColorToTrainingsset(float* color, int labelID)
{
	if(labelID > labelCount-1)
	{
		return;
	}

	if(trainingsSetCount[labelID] == numColorsPerLabel)
	{
		numColorsPerLabel *= 2;
		float*** newTrainingsSet = (float***)std::malloc(sizeof(float**) * labelCount);
		for (int i = 0; i < labelCount; ++i)
		{
			newTrainingsSet[i] = (float**)malloc(sizeof(float*) * numColorsPerLabel);
			for (int j = 0; j < trainingsSetCount[i]; ++j)
			{
				newTrainingsSet[i][j] = trainingsSet[i][j];
			}
			free(trainingsSet[i]);
		}
		free(trainingsSet);

		trainingsSet = newTrainingsSet;
	}

	trainingsSet[labelID][trainingsSetCount[labelID]] = color;
	trainingsSetCount[labelID]++;
}

void KNN::AddColorsToTrainingsset(float** colors, int labelID, int n)
{
	if (labelID > labelCount - 1)
	{
		return;
	}

	int newColorPerLabelCount = numColorsPerLabel;

	while(trainingsSetCount[labelID] + n > newColorPerLabelCount)
	{
		newColorPerLabelCount *= 2;
	}

	if (newColorPerLabelCount > numColorsPerLabel)
	{
		numColorsPerLabel *= newColorPerLabelCount;
		float*** newTrainingsSet = (float***)std::malloc(sizeof(float**) * labelCount);
		for (int i = 0; i < labelCount; ++i)
		{
			newTrainingsSet[i] = (float**)malloc(sizeof(float*) * numColorsPerLabel);
			for (int j = 0; j < trainingsSetCount[i]; ++j)
			{
				newTrainingsSet[i][j] = trainingsSet[i][j];
			}
			free(trainingsSet[i]);
		}

		free(trainingsSet);
		trainingsSet = newTrainingsSet;
	}

	for(int i = 0; i < n; ++i)
	{
		trainingsSet[labelID][trainingsSetCount[labelID]] = colors[n];
		trainingsSetCount[labelID]++;
	}
}