#pragma once
#include "RgbLab.h"
#include <iostream>

void main(){

	float whitergb[3] = { 1.0, 1.0, 1.0 };
	float* white = &whitergb[0];

	float* labWhite = RgbLab::RgbToLab(white);

	std::cout << labWhite[0] << ", " << labWhite[1] << ", " << labWhite[2] << std::endl;

	float * rgbWhite = RgbLab::LabToRgb(labWhite);

	std::cout << rgbWhite[0] << ", " << rgbWhite[1] << ", " << rgbWhite[2] << std::endl;

	char a;
	std::cin >> a;

	free(labWhite);
	free(rgbWhite);
}
