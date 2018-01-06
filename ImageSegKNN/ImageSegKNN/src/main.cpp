#pragma once
#include "header/cuda_RgbLab.cuh"
#include "header/KNN.h"
#include "header/ppma_io.hpp"
#include <iostream>

KNN<3>* DoctorTestKNN()
{
	float** labelColors = (float**)malloc(sizeof(float*) * 3);
	float* red = RgbLab::MakeColor(1, 0, 0);
	float* green = RgbLab::MakeColor(0, 1, 0);
	float* blue = RgbLab::MakeColor(0, 0, 1);

	labelColors[0] = red;
	labelColors[1] = green;
	labelColors[2] = blue;

	int numColorsPerLabel = 8;

	float*** trainingsSet = (float***)std::malloc(sizeof(float**) * 3);
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

void main(){

	std::string file = "images/DoctorTest.ppm";
	int xsize = 0;
	int ysize = 0;
	int maxrgb = 0;

	KNN<3> knn = *DoctorTestKNN();
	
	int** r = (int**) malloc(sizeof(int*));
	int** g = (int**) malloc(sizeof(int*));
	int** b = (int**) malloc(sizeof(int*));

	ppma_read(file, xsize, ysize, maxrgb, r, g, b);

	int* rnew = (int*) malloc(sizeof(int) * xsize * ysize);
	int* gnew = (int*) malloc(sizeof(int) * xsize * ysize);
	int* bnew = (int*) malloc(sizeof(int) * xsize * ysize);
	
	for(int i = 0; i < xsize*ysize; ++i)
	{
		float color[3] = { r[0][i] / 255.0f, g[0][i] / 255.0f, b[0][i] / 255.0f };
		float* labelcolor = knn.GetLabelColor(knn.DetermineLabelRgb(6, &color[0], true));
		rnew[i] = labelcolor[0] * 255;
		gnew[i] = labelcolor[1] * 255;
		bnew[i] = labelcolor[2] * 255;
	}

	ppma_write("images/DoctorTestSeg.ppm", xsize, ysize, rnew, gnew, bnew);

	char a;
	std::cin >> a;

	free(*r);
	free(r);
	free(*g);
	free(g);
	free(*b);
	free(b);
	free(rnew);
	free(gnew);
	free(bnew);
}
