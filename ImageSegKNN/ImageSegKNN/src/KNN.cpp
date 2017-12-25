#pragma once
#include "header/KNN.h"
#include <cstdlib>
#include "header/cuda_RgbLab.cuh"


KNN::KNN() : labelCount(2) 
{
	//  allocating memory and initializing data

	numColorsPerLabel = 5;
	
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

	numColorsPerLabel = 5;

	trainingsSet = (float***) std::malloc(sizeof(float) * labelCount * numColorsPerLabel * 3);
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

// naiive implementation of finding the k nearest neighbours
// any other implementation might be better

int KNN::DetermineLabel(int k, float* data, bool weighted)
{
	float** neighbours = (float**) malloc(sizeof(float) * k * 3);
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
	for(int i = 0; i < labelCount; ++i)
	{
		for(int j = 0; j < trainingsSetCount[i]; ++j)
		{
			// get the distance
			float length = RgbLab::ColorDistance(data, trainingsSet[i][j], 3);
			for(int l = 0; l < k; ++l)
			{
				// if we find something closer than the latest k nearest
				// update our list
				if(length < distances[l])
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

	for(int i = 0; i < labelCount; ++i)
	{
		if(maxVote < voteCount[i])
		{
			maxVote = voteCount[i];
			winningLabel = i;
		}
	}

	free(neighbours);
	free(distances);
	free(labels);
	free(voteCount);

	return winningLabel;
}

void KNN::AddColorToTrainingsset(float* color, int labelID)
{
	//TODO implement
}

void KNN::AddColorsToTrainingsset(float** colors, int labelID)
{
	//TODO implement
}