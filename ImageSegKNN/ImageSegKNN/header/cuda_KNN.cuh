#pragma once
#include <crt/host_defines.h>
#include "header/cuda_RgbLab.cuh"

struct NeighbourEntry
{
	float distance;
	int label;

	__host__ __device__ NeighbourEntry() : distance(0), label(0)
	{
	}

	__host__ __device__ NeighbourEntry(float distance, int label) : distance(distance), label(label)
	{
	}

	__host__ __device__ bool operator< (const NeighbourEntry& rhs) const { return this->distance < rhs.distance; }
	__host__ __device__ bool operator> (const NeighbourEntry& rhs) const { return rhs < *this; }
	__host__ __device__ bool operator<= (const NeighbourEntry& rhs) const { return !(*this > rhs); }
	__host__ __device__ bool operator>= (const NeighbourEntry& rhs) const { return !(*this < rhs); }

	__host__ __device__ bool operator== (const NeighbourEntry& rhs) const { return this->distance == rhs.distance; }
	__host__ __device__ bool operator!= (const NeighbourEntry& rhs) const { return !(*this == rhs); }
};

//number of labels
template <int labelCount>
class KNN
{
public:
	__host__ KNN();
	__host__ KNN(float* labelColors);
	__host__ KNN(float* labelColors, float* trainingsSet, int* trainingsSetCount,
		int numColorsPerLabel);
	__host__ ~KNN();

	__host__ void AddColorToTrainingsset(float* color, int labelID);
	__host__ void AddColorsToTrainingsset(float** colors, int labelID, int n);

	__host__ __device__ float* GetLabelColor(int labelID);

	__host__ __device__ int DetermineLabelRgb(int k, float* data, bool weighted);
	__host__ __device__ int DetermineLabelLab(int k, float* data, bool weighted);

	// no-heap implementation
	//uses insertion sort
	template <int k>
	__host__ __device__ int DetermineLabelRgb(float* data, bool weighted)
	{
		NeighbourEntry neighboursEntry[k];
		float voteCount[labelCount];

		//avoid false results by using at max the amount of trainingsentries
		int newk = k;
		if (trainingEntriesCount < newk)
		{
			newk = trainingEntriesCount;
		}

		// initialize allocated memory;
		for (int i = 0; i < labelCount; ++i)
		{
			voteCount[i] = 0;
		}

		//initial votes will be removed while the entries get updated
		voteCount[0] = newk * 1.0f / 10000.0f;

		for (int i = 0; i < newk; ++i)
		{
			neighboursEntry[i] = NeighbourEntry(10000, 0);
		}

		// go through each trainingsdata
		for (int i = 0; i < labelCount; ++i)
		{
			for (int j = 0; j < trainingsSetCount[i]; ++j)
			{
				// get the distance
				float length = RgbLab::ColorDistance(data, getColorInTrainingsset(i, j));
				for (int l = 0; l < newk; ++l)
				{
					// if we find something closer than the latest k nearest
					// update our list
					if (length < neighboursEntry[l].distance)
					{
						// update the votes
						if (weighted)
						{
							voteCount[i] += 1.0 / (length + 0.0000001);
							voteCount[neighboursEntry[newk - 1].label] -= 1.0 / neighboursEntry[newk - 1].distance;
						}
						else
						{
							voteCount[i] += 1.0;
							voteCount[neighboursEntry[newk - 1].label] -= 1.0;
						}

						// therefore we have to insert the new and push the ones behind it one index further (aka copy them)
						for (int m = newk - 2; m > l; --m)
						{
							neighboursEntry[m + 1] = neighboursEntry[m];
						}

						neighboursEntry[l] = NeighbourEntry((length + 0.0000001), i);

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

		return winningLabel;
	}

	template <int k>
	__host__ __device__ int DetermineLabelLab(float* data, bool weighted)
	{
		NeighbourEntry neighboursEntry[k];
		float voteCount[labelCount];

		//avoid false results by using at max the amount of trainingsentries
		int newk = k;
		if (trainingEntriesCount < newk)
		{
			newk = trainingEntriesCount;
		}

		// initialize allocated memory;
		for (int i = 0; i < labelCount; ++i)
		{
			voteCount[i] = 0;
		}

		//initial votes will be removed while the entries get updated
		voteCount[0] = newk * 1.0f / 10000.0f;

		for (int i = 0; i < newk; ++i)
		{
			neighboursEntry[i] = NeighbourEntry(10000, 0);
		}

		float* datalab = &RgbLab::RgbToLab(data).color[0];

		// go through each trainingsdata
		for (int i = 0; i < labelCount; ++i)
		{
			for (int j = 0; j < trainingsSetCount[i]; ++j)
			{
				float* tslab = &RgbLab::RgbToLab(getColorInTrainingsset(i, j)).color[0];

				// get the distance
				float length = RgbLab::ColorDistance(datalab, tslab);
				for (int l = 0; l < newk; ++l)
				{
					// if we find something closer than the latest k nearest
					// update our list
					if (length < neighboursEntry[l].distance)
					{
						// update the votes
						if (weighted)
						{
							voteCount[i] += 1.0 / (length+0.0000001);
							voteCount[neighboursEntry[newk - 1].label] -= 1.0 / neighboursEntry[newk - 1].distance;
						}
						else
						{
							voteCount[i] += 1.0;
							voteCount[neighboursEntry[newk - 1].label] -= 1.0;
						}

						// therefore we have to insert the new and push the ones behind it one index further (aka copy them)
						for (int m = newk - 2; m > l; --m)
						{
							neighboursEntry[m + 1] = neighboursEntry[m];
						}

						neighboursEntry[l] = NeighbourEntry((length + 0.0000001), i);

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

		return winningLabel;
	}

	__host__ __device__ float* GetLabelColors()
	{
		return labelColors;
	}

	__host__ __device__ float* GetTrainingsSet()
	{
		return trainingsSet;
	}

	__host__ __device__ int* GetTrainingsSetCount()
	{
		return trainingsSetCount;
	}

	__host__ __device__ int GetNumColorsPerLabel()
	{
		return numColorsPerLabel;
	}

	int GetSizeOfLabelColors()
	{
		return sizeof(float) * labelCount * 3;
	}

	int GetSizeOfTrainingsset()
	{
		return sizeof(float) * labelCount * numColorsPerLabel * 3;
	}

	int GetSizeOfTrainingsSetCount()
	{
		return  sizeof(int) * labelCount;
	}

	int GetNumTrainingsEntries()
	{
		return trainingEntriesCount;
	}

private:
	// color which each label will be assigned to
	float* labelColors;

	//max number of training set entries we can have
	static const int maxNumberTrainingEntries = 200;

	//current number of all training set entries wie currently have
	int trainingEntriesCount;

	// keep track how many colors we have in each label as trainingsdata
	int* trainingsSetCount;

	// need to keep trainingsset dynamic size, but cuda only allows c-type things -> have to keep track how many colors
	// we can have per label and eventually reallocate
	int numColorsPerLabel;

	//array containing trainingsdata fpr the labels
	float* trainingsSet;

	//get the r - value of the color in the trainingsset
	float* getColorInTrainingsset(int label, int index)
	{
		return &trainingsSet[label * numColorsPerLabel * 3 + index * 3];
	}

	//get the index of the r - value of a color in the trainingsset
	int getIndexInTrainingssset(int label, int index)
	{
		return label * numColorsPerLabel * 3 + index * 3;
	}

	//write color to the trainingsset
	void assignColorToTrainingsset(float* color, int label, int index)
	{
		int i = getIndexInTrainingssset(label, index);
		for(int j = 0; j < 3; ++j)
		{
			trainingsSet[i + j] = color[j];
		}
	}

	//write color in the labelcolors array
	void assignColorToLabel(float* color, int label)
	{
		int i = label * 3;
		for (int j = 0; j < 3; ++j)
		{
			labelColors[i + j] = color[j];
		}
	}
};