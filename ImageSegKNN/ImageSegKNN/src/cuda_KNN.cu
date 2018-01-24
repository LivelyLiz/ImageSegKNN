#pragma once
#include "header/cuda_KNN.cuh"
#include <cstdlib>
#include "header/cuda_RgbLab.cuh"
#include "header/cuda_MinHeap.cuh"

//only possible with 2 labels
KNN<2>::KNN() : trainingEntriesCount(2), numColorsPerLabel(8)
{
	//  allocating memory and initializing data

	int labelCount = 2;

	trainingsSet = (float*)std::malloc(sizeof(float) * labelCount * numColorsPerLabel * 3);

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

	assignColorToTrainingsset(&white[0], 0, 0);
	assignColorToTrainingsset(&black[0], 1, 0);

	labelColors = (float*)malloc(sizeof(float) * labelCount * 3);
	assignColorToLabel(&blue[0], 0);
	assignColorToLabel(&red[0], 1);
}

template <int labelCount>
KNN<labelCount>::KNN(float* labelColors) : labelColors(labelColors), trainingEntriesCount(0)
{
	trainingsSetCount = (int*)std::malloc(sizeof(float) * labelCount);
	for(int i = 0; i < labelCount; ++i)
	{
		trainingsSetCount[i] = 0;
	}

	numColorsPerLabel = 8;

	trainingsSet = (float*)std::malloc(sizeof(float) * labelCount * numColorsPerLabel * 3);
}

template <int labelCount>
KNN<labelCount>::KNN(float* labelColors, float* ts, int* trainingsSetCount, int maxColorsInLabel) : labelColors(labelColors), trainingsSet(ts),
trainingsSetCount(trainingsSetCount), numColorsPerLabel(maxColorsInLabel), trainingEntriesCount(0)
{
	for (int i = 0; i < labelCount; ++i)
	{
		trainingEntriesCount += trainingsSetCount[i];
	}
}

template <int labelCount>
KNN<labelCount>::~KNN()
{
	free(trainingsSet);
	free(trainingsSetCount);
	free(labelColors);
}

template <int labelCount>
float* KNN<labelCount>::GetLabelColor(int labelID)
{
	if (labelID < labelCount)
	{
		return &labelColors[labelID*3];
	}
	return nullptr;
}

template <int labelCount>
int KNN<labelCount>::DetermineLabelLab(int k, float* data, bool weighted)
{
	MinHeap<NeighbourEntry, maxNumberTrainingEntries> heap;

	float* datalab = &RgbLab::RgbToLab(data).color[0];

	// go through each of the trainingsdata
	for (int i = 0; i < labelCount; ++i)
	{
		for (int j = 0; j < trainingsSetCount[i]; ++j)
		{
			float* tslab = &RgbLab::RgbToLab(getColorInTrainingsset(i, j)).color[0];

			// get the distance
			float length = RgbLab::ColorDistance(datalab, tslab);

			heap.Insert(NeighbourEntry(length, i));
		}
	}

	float voteCount[labelCount];
	for (int i = 0; i < labelCount; ++i)
	{
		voteCount[i] = 0;
	}

	//avoid false results by using at max the amount of trainingsentries
	int newk = k;
	if (trainingEntriesCount < newk)
	{
		newk = trainingEntriesCount;
	}

	//get the k smallest elements
	for (int i = 0; i < newk; ++i)
	{
		NeighbourEntry ne = heap.Pop();
		if (weighted)
		{
			voteCount[ne.label] += 1.0f / (ne.distance + 0.0000001);
		}
		else
		{
			voteCount[ne.label] += 1.0f;
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

	return winningLabel;
}

// minheap implementation
template <int labelCount>
int KNN<labelCount>::DetermineLabelRgb(int k, float* data, bool weighted)
{
	MinHeap<NeighbourEntry, maxNumberTrainingEntries> heap;

	// go through each trainingsdata
	for (int i = 0; i < labelCount; ++i)
	{
		for (int j = 0; j < trainingsSetCount[i]; ++j)
		{
			// get the distance
			float length = RgbLab::ColorDistance(data, getColorInTrainingsset(i, j));

			heap.Insert(NeighbourEntry(length, i));
		}
	}

	float voteCount[labelCount];
	for (int i = 0; i < labelCount; ++i)
	{
		voteCount[i] = 0;
	}

	//avoid false results by using at max the amount of trainingsentries
	int newk = k;
	if (trainingEntriesCount < newk)
	{
		newk = trainingEntriesCount;
	}

	for (int i = 0; i < newk; ++i)
	{
		NeighbourEntry ne = heap.Pop();
		if (weighted)
		{
			voteCount[ne.label] += 1.0f / (ne.distance + 0.0000001);
		}
		else
		{
			voteCount[ne.label] += 1.0f;
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

	return winningLabel;
}

template<int labelCount>
void KNN<labelCount>::AddColorToTrainingsset(float* color, int labelID)
{
	//if invalid label given or number of trainingsentries is already at max
	if (labelID > labelCount - 1 || trainingEntriesCount == maxNumberTrainingEntries)
	{
		return;
	}

	//if not enough memory free, allocate more and copy each color to the new
	if (trainingsSetCount[labelID] == numColorsPerLabel)
	{
		numColorsPerLabel *= 2;
		float* newTrainingsSet = (float*)std::malloc(sizeof(float) * labelCount * numColorsPerLabel * 3);
		for (int i = 0; i < labelCount; ++i)
		{
			for (int j = 0; j < trainingsSetCount[i]; ++j)
			{
				newTrainingsSet[getIndexInTrainingssset(i,j)] = trainingsSet[i * numColorsPerLabel/2 * 3 + j * 3];
			}
		}
		free(trainingsSet);

		trainingsSet = newTrainingsSet;
	}

	//add the actual new value
	assignColorToTrainingsset(color, labelID, trainingsSetCount[labelID]);
	trainingsSetCount[labelID]++;
	trainingEntriesCount++;
}

template<int labelCount>
void KNN<labelCount>::AddColorsToTrainingsset(float** colors, int labelID, int n)
{
	//if invalid label is given or number of trainingsdata is at max
	if (labelID > labelCount - 1 || trainingEntriesCount == maxNumberTrainingEntries)
	{
		return;
	}

	//if there is still capacity, but not enough for all, just add the first
	if (trainingEntriesCount + n > maxNumberTrainingEntries)
	{
		n = maxNumberTrainingEntries - trainingEntriesCount;
	}

	//determine how much memory will be needed
	int newColorPerLabelCount = numColorsPerLabel;

	while (trainingsSetCount[labelID] + n > newColorPerLabelCount)
	{
		newColorPerLabelCount *= 2;
	}

	//allocate new memory and copy old data
	if (newColorPerLabelCount > numColorsPerLabel)
	{
		
		float* newTrainingsSet = (float*)std::malloc(sizeof(float) * labelCount * newColorPerLabelCount * 3);
		for (int i = 0; i < labelCount; ++i)
		{
			for (int j = 0; j < trainingsSetCount[i]; ++j)
			{
				newTrainingsSet[i * newColorPerLabelCount / 2 * 3 + j * 3] = trainingsSet[getIndexInTrainingssset(i, j)];
			}
		}
		numColorsPerLabel = newColorPerLabelCount;
		free(trainingsSet);
		trainingsSet = newTrainingsSet;
	}

	//actually add the new entries
	for (int i = 0; i < n; ++i)
	{
		assignColorToTrainingsset(colors[i], labelID, trainingsSetCount[labelID]);
		trainingsSetCount[labelID]++;
		trainingEntriesCount++;
	}
}

//explicit template instantiation needed for linking
template class KNN<2>;
template class KNN<3>;
template class KNN<4>;
template class KNN<5>;
template class KNN<6>;
template class KNN<7>;
template class KNN<8>;
template class KNN<9>;