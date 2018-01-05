/*#include "header/cuda_RgbLab.cuh";

#ifdef __CUDA_ARCH__
	// conversion matrices for sRGB to Lab with D50 white
	float RgbLab::RgbToXyzMatrix[3][3] = { { 0.4360747,  0.3850649,  0.1430804 },
	{ 0.2225045,  0.7168786,  0.0606169 },
	{ 0.0139322,  0.0971045,  0.7141733 } };

float RgbLab::XyzToRgbMatrix[3][3] = { { 3.1338561, -1.6168667, -0.4906146 },
{ -0.9787684,  1.9161415,  0.0334540 },
{ 0.0719453, -0.2289914,  1.4052427 } };

//reference white D50
float RgbLab::XyzReferenceWhite[3] = { 0.9642, 1.0000, 0.8251 };

float RgbLab::delta = 6. / 29.;
#else
// conversion matrices for sRGB to Lab with D50 white
float const RgbLab::RgbToXyzMatrix[3][3] = { { 0.4360747,  0.3850649,  0.1430804 },
{ 0.2225045,  0.7168786,  0.0606169 },
{ 0.0139322,  0.0971045,  0.7141733 } };

float const RgbLab::XyzToRgbMatrix[3][3] = { { 3.1338561, -1.6168667, -0.4906146 },
{ -0.9787684,  1.9161415,  0.0334540 },
{ 0.0719453, -0.2289914,  1.4052427 } };

//reference white D50
float const RgbLab::XyzReferenceWhite[3] = { 0.9642, 1.0000, 0.8251 };

float const RgbLab::delta = 6. / 29.;
#endif
*/