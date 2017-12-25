#pragma once
#include "header/cuda_RgbLab.cuh"
#include "header/KNN.h"
#include <iostream>

void main(){

	// load image with stb_image will result in an char* array?

	KNN knn;

	float whitergb[3] = { 1.0, 1.0, 1.0 };
	float blackrgb[3] = { 0.0, 0.0, 0.0 };
	float fairgreyrgb[3] = { 0.8, 0.8, 0.8 };
	float greyrgb[3] = { 0.5, 0.5, 0.5 };
	float darkgreyrgb[3] = { 0.2, 0.2, 0.2 };

	float* white = &whitergb[0];
	float* black = &blackrgb[0];
	float* fairgrey = &fairgreyrgb[0];
	float* grey = &greyrgb[0];
	float* darkgrey = &darkgreyrgb[0];

	RgbLab::RgbToLab(white);

	std::cout << white[0] << ", " << white[1] << ", " << white[2] << std::endl;

	RgbLab::LabToRgb(white);

	std::cout << white[0] << ", " << white[1] << ", " << white[2] << std::endl;

	int label = knn.DetermineLabel(1, fairgrey, true);

	std::cout << label << std::endl;

	char a;
	std::cin >> a;
}
