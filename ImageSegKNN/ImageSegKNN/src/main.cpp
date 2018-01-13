#pragma once
#include "header/cuda_RgbLab.cuh"
#include "header/cuda_KNN.cuh"
#include "header/ppma_io.hpp"
#include <iostream>

/*KNN<3>* DoctorTestKNN()
{
	float** labelColors = (float**)malloc(sizeof(float*) * 3);
	float* red = RgbLab::MakeColor(1, 0, 0);
	float* green = RgbLab::MakeColor(0, 1, 0);
	float* blue = RgbLab::MakeColor(0, 0, 1);

	labelColors[0] = red;
	labelColors[1] = green;
	labelColors[2] = blue;

	int numColorsPerLabel = 8;

	float*** trainingsSet = (float***)malloc(sizeof(float**) * 3);
	for (int i = 0; i < 3; ++i)
	{
		trainingsSet[i] = (float**)malloc(sizeof(float*) * numColorsPerLabel);
	}

	float** label1colors = (float**)malloc(sizeof(float*) * numColorsPerLabel);
	float** label2colors = (float**)malloc(sizeof(float*) * numColorsPerLabel);
	float** label3colors = (float**)malloc(sizeof(float*) * numColorsPerLabel);

	label1colors[0] = RgbLab::MakeColor(238, 112, 54);
	label1colors[1] = RgbLab::MakeColor(250, 169, 175);
	label1colors[2] = RgbLab::MakeColor(245, 145, 119);
	label1colors[3] = RgbLab::MakeColor(253, 180, 125);

	label2colors[0] = RgbLab::MakeColor(216, 143, 92);
	label2colors[1] = RgbLab::MakeColor(227, 157, 108);
	label2colors[2] = RgbLab::MakeColor(173, 125, 89);
	label2colors[3] = RgbLab::MakeColor(248, 194, 150);

	label3colors[0] = RgbLab::MakeColor(152, 43, 49);
	label3colors[1] = RgbLab::MakeColor(68, 58, 56);
	label3colors[2] = RgbLab::MakeColor(7, 6, 4);
	label3colors[3] = RgbLab::MakeColor(117, 74, 58);

	trainingsSet[0] = label1colors;
	trainingsSet[1] = label2colors;
	trainingsSet[2] = label3colors;

	int* trainingsSetCount = (int*)malloc(sizeof(int) * 3);
	trainingsSetCount[0] = 4;
	trainingsSetCount[1] = 4;
	trainingsSetCount[2] = 4;
		

	KNN<3>* knn = new KNN<3>(labelColors, trainingsSet, trainingsSetCount, numColorsPerLabel);
	return knn;
}

KNN<3>* MirrorTestKNN()
{
	float* labelColors = (float*)malloc(sizeof(float) * 3 * 3);
	float* red = RgbLab::MakeColor(255, 0, 0);
	float* green = RgbLab::MakeColor(0, 255, 0);
	float* blue = RgbLab::MakeColor(0, 0, 255);

	for(int i = 0; i < 3; ++i)
	{
		labelColors[0*3 + i] = red[i];
		labelColors[1*3 + i] = green[i];
		labelColors[2*3 + i] = blue[i];
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

void main(){

	std::string file = "images/MirrorsEdgeTest.ppm";
	int xsize = 0;
	int ysize = 0;
	int maxrgb = 0;

	KNN<3> knn = *MirrorTestKNN();

	int* r;
	int* g;
	int* b;

	ppma_read(file, xsize, ysize, maxrgb, &r, &g, &b);

	int numPixels = xsize * ysize;

	float* host_picturedata = (float*)malloc(sizeof(float) * numPixels * 3);
	for (int i = 0; i < numPixels; ++i)
	{
		host_picturedata[3 * i] = r[i] / 255.0f;
		host_picturedata[3 * i + 1] = g[i] / 255.0f;
		host_picturedata[3 * i + 2] = b[i] / 255.0f;
	}

	int* rnew = (int*) malloc(sizeof(int) * xsize * ysize);
	int* gnew = (int*) malloc(sizeof(int) * xsize * ysize);
	int* bnew = (int*) malloc(sizeof(int) * xsize * ysize);
	
	for(int i = 0; i < xsize*ysize; ++i)
	{
		float color[3] = { host_picturedata[i*3], host_picturedata[i*3+1], host_picturedata[i*3 + 2]};
		float* labelcolor = knn.GetLabelColor(knn.DetermineLabelLab(7, &color[0], true));
		rnew[i] = (int) (labelcolor[0] * 255);
		gnew[i] = (int) (labelcolor[1] * 255);
		bnew[i] = (int) (labelcolor[2] * 255);
	}

	ppma_write("images/MirrorsEdgeTestSeg.ppm", xsize, ysize, rnew, gnew, bnew);

	printf("ready");

	char a;
	std::cin >> a;

	free(r);
	free(g);
	free(b);
	free(rnew);
	free(gnew);
	free(bnew);
	free(host_picturedata);
}
*/