// lots and lots of pointers and stuff, we might want to change that? But we need this on cuda device, so it's difficult

class KNN
{
public:
	KNN();
	KNN(int labelCount, float** labelColors);
	KNN(int labelCount, float** labelColors, float*** trainingsSet, int* trainingsSetCount,
		int numColorsPerLabel);
	~KNN();

	void AddColorToTrainingsset(float* color, int labelID);
	void AddColorsToTrainingsset(float** colors, int labelID);
	
	float* GetLabelColor(int labelID);

	int DetermineLabel(int k, float* data, bool weighted);

private:
	// number of labels
	const int labelCount;
	
	// color which each label will be assigned to
	float** labelColors;


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
};