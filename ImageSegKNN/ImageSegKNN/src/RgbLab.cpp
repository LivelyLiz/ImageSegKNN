/*#include "header/cuda_RgbLab.cuh"

// conversion matrices for sRGB to Lab with D50 white
float const RgbLab::RgbToXyzMatrix[3][3] = { { 0.4360747f,  0.3850649f,  0.1430804f},
{0.2225045f,  0.7168786f,  0.0606169f},
{0.0139322f,  0.0971045f,  0.7141733f } };

float const RgbLab::XyzToRgbMatrix[3][3] = { {3.1338561f, -1.6168667f, -0.4906146f },
{- 0.9787684f,  1.9161415f,  0.0334540f },
{0.0719453f, -0.2289914f,  1.4052427f } };

//reference white D50
float const RgbLab::XyzReferenceWhite[3] = { 0.9642f, 1.0000f, 0.8251f };

float const RgbLab::delta = 6.0f / 29.0f;*/