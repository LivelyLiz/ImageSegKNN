// lots and lots of pointers and stuff, we might want to change that? But we need this on cuda device, so it's difficult
#pragma once

//number of labels
template <int labelCount>
class KNN
{
public:
	KNN();
	KNN(float** labelColors);
	KNN(float** labelColors, float*** trainingsSet, int* trainingsSetCount,
		int numColorsPerLabel);
	~KNN();

	void AddColorToTrainingsset(float* color, int labelID);
	void AddColorsToTrainingsset(float** colors, int labelID, int n);
	
	float* GetLabelColor(int labelID);

	int DetermineLabelRgb(int k, float* data, bool weighted);
	int DetermineLabelLab(int k, float* data, bool weighted);

	// no-heap implementation 
	// using a non type template function enables compile time optimization
	// and it is also possible to allocate memory on the stack instead of heap
	// in CUDA this means it is at least possible to have the arrays in the registers
	// instead of global memory -> performance improvement!!
	template <int k>
	int DetermineLabelRgb(float* data, bool weighted)
	{
		NeighbourEntry neighboursEntry[k];
		float voteCount[labelCount];

		// initialize allocated memory;
		for (int i = 0; i < labelCount; ++i)
		{
			voteCount[i] = 0;
		}

		for (int i = 0; i < k; ++i)
		{
			neighboursEntry[i] = NeighbourEntry(10000, 0);
		}

		// go through each trainingsdata
		for (int i = 0; i < labelCount; ++i)
		{
			for (int j = 0; j < trainingsSetCount[i]; ++j)
			{
				// get the distance
				float length = RgbLab::ColorDistance(data, trainingsSet[i][j]);
				for (int l = 0; l < k; ++l)
				{
					// if we find something closer than the latest k nearest
					// update our list
					if (length < neighboursEntry[l].distance)
					{
						// update the votes
						if (weighted)
						{
							voteCount[i] += 1.0f / length;
							voteCount[neighboursEntry[k - 1].label] -= 1.0f / neighboursEntry[k - 1].distance;
						}
						else
						{
							voteCount[i] += 1.0f;
							voteCount[neighboursEntry[k - 1].label] -= 1.0f;
						}

						// therefore we have to insert the new and push the ones behind it one index further (aka copy them)
						for (int m = k - 2; m > l; --m)
						{
							neighboursEntry[m + 1] = neighboursEntry[m];
						}

						neighboursEntry[l] = NeighbourEntry(length, i);

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
	int DetermineLabelLab(float* data, bool weighted)
	{
		NeighbourEntry neighboursEntry[k];
		float voteCount[labelCount];

		// initialize allocated memory;
		for (int i = 0; i < labelCount; ++i)
		{
			voteCount[i] = 0;
		}

		for (int i = 0; i < k; ++i)
		{
			neighboursEntry[i] = NeighbourEntry(10000, 0);
		}

		float* datalab = &RgbLab::RgbToLab(data).color[0];

		// go through each trainingsdata
		for (int i = 0; i < labelCount; ++i)
		{
			for (int j = 0; j < trainingsSetCount[i]; ++j)
			{
				float* tslab = &RgbLab::RgbToLab(trainingsSet[i][j]).color[0];

				// get the distance
				float length = RgbLab::ColorDistance(datalab, tslab);
				for (int l = 0; l < k; ++l)
				{
					// if we find something closer than the latest k nearest
					// update our list
					if (length < neighboursEntry[l].distance)
					{
						// update the votes
						if (weighted)
						{
							voteCount[i] += 1.0 / length;
							voteCount[neighboursEntry[k - 1].label] -= 1.0 / neighboursEntry[k - 1].distance;
						}
						else
						{
							voteCount[i] += 1.0;
							voteCount[neighboursEntry[k - 1].label] -= 1.0;
						}

						// therefore we have to insert the new and push the ones behind it one index further (aka copy them)
						for (int m = k - 2; m > l; --m)
						{
							neighboursEntry[m + 1] = neighboursEntry[m];
						}

						neighboursEntry[l] = NeighbourEntry(length, i);

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

private:
	// color which each label will be assigned to
	float** labelColors;

	//max number of training set entries we can have
	static const int maxNumberTrainingEntries = 200;

	//current number of all training set entries wie currently have
	int trainingEntriesCount;

	// keep track how many colors we have in each label as trainingsdata
	int* trainingsSetCount;
	
	// need to keep trainingsset dynamic size, but cuda only allows c-type things -> have to keep track how many colors
	// we can have per label and eventually reallocate
	int numColorsPerLabel;

	// this is an array (pointer) to number of labels arrays (pointer) 
	// in which are the colors belonging to the trainingSet (colors are pointers too)
	// example with 2 labels
	// [ [[0,0,0], [0,0,0.2]],  [[1,1,1], [1, 0.9, 0.9]] ]
	// so for the first label, the second color -> float* color = trainingsSet[0][1]
	float*** trainingsSet;

	struct NeighbourEntry
	{
		float distance;
		int label;

		NeighbourEntry() : distance(0), label(0)
		{
		}

		NeighbourEntry(float distance, int label) : distance(distance), label(label)
		{
		}

		bool operator< (const NeighbourEntry& rhs) const { return this->distance < rhs.distance; }
		bool operator> (const NeighbourEntry& rhs) const { return rhs < *this; }
		bool operator<= ( const NeighbourEntry& rhs) const { return !(*this > rhs); }
		bool operator>= (const NeighbourEntry& rhs) const { return !(*this < rhs); }

		bool operator== (const NeighbourEntry& rhs) const { return this->distance == rhs.distance; }
		bool operator!= (const NeighbourEntry& rhs) const { return !(*this == rhs); }
	};
};