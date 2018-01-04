#pragma once
#include "header/cuda_RgbLab.cuh"
#include "header/KNN.h"
#include "header/ppma_io.hpp"
#include <iostream>

void main(){

	std::string file = "images/BlackWhiteTest.ppm";
	int xsize = 200;
	int ysize = 200;
	int maxrgb = 255;

	KNN<2> knn;
	int** r = (int**) malloc(sizeof(int*));
	int** g = (int**) malloc(sizeof(int*));
	int** b = (int**) malloc(sizeof(int*));

	int* rnew = (int*) malloc(sizeof(int) * xsize * ysize);
	int* gnew = (int*) malloc(sizeof(int) * xsize * ysize);
	int* bnew = (int*) malloc(sizeof(int) * xsize * ysize);

	ppma_read(file, xsize, ysize, maxrgb, r, g, b);
	
	for(int i = 0; i < xsize*ysize; ++i)
	{
		float color[3] = { r[0][i] / 255.0, g[0][i] / 255.0, b[0][i] / 255.0 };
		float* labelcolor = knn.GetLabelColor(knn.DetermineLabelRgb<2>(&color[0], true));
		rnew[i] = labelcolor[0] * 255;
		gnew[i] = labelcolor[1] * 255;
		bnew[i] = labelcolor[2] * 255;
	}

	ppma_write("images/BlackWhiteTestSeg.ppm", xsize, ysize, rnew, gnew, bnew);

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
