#pragma once
#include <cstdlib>
#include <crt/host_defines.h>
#include <math.h>

// Math from http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html and https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
class RgbLab
{
public:
	struct Color
	{
		float color[3];
	};

	//Converts a color in sRGB (values [0.0, 1.0]) to the Lab color space (values [0.0, 1.0])
	__host__ __device__ static Color RgbToLab(float* rgbVal)
	{
		float RgbToXyzMatrix[3][3] = { { 0.4360747f,  0.3850649f,  0.1430804f },
		{ 0.2225045f,  0.7168786f,  0.0606169f },
		{ 0.0139322f,  0.0971045f,  0.7141733f } };

		float XyzReferenceWhite[3] = { 0.9642f, 1.0000f, 0.8251f };

		Color lab;

		float X = RgbToXyzMatrix[0][0] * rgbVal[0] + RgbToXyzMatrix[0][1] * rgbVal[1] + RgbToXyzMatrix[0][2] * rgbVal[2];
		float Y = RgbToXyzMatrix[1][0] * rgbVal[0] + RgbToXyzMatrix[1][1] * rgbVal[1] + RgbToXyzMatrix[1][2] * rgbVal[2];
		float Z = RgbToXyzMatrix[2][0] * rgbVal[0] + RgbToXyzMatrix[2][1] * rgbVal[1] + RgbToXyzMatrix[2][2] * rgbVal[2];

		float L = 1.16f * f(Y / XyzReferenceWhite[1]) - .16f;
		float a = 5.00f * (f(X / XyzReferenceWhite[0]) - f(Y / XyzReferenceWhite[1]));
		float b = 2.00f * (f(Y / XyzReferenceWhite[1]) - f(Z / XyzReferenceWhite[2]));

		lab.color[0] = L;
		lab.color[1] = a;
		lab.color[2] = b;

		return lab;
	}

	//Converts a color in Lab color space (values [0.0, 1.0]) to the sRGB (values [0.0, 1.0])
	__host__ __device__ static Color LabToRgb(float* labVal)
	{
		float XyzToRgbMatrix[3][3] = { { 3.1338561f, -1.6168667f, -0.4906146f },
		{ -0.9787684f,  1.9161415f,  0.0334540f },
		{ 0.0719453f, -0.2289914f,  1.4052427f } };

		float XyzReferenceWhite[3] = { 0.9642f, 1.0000f, 0.8251f };

		Color rgb;

		float X = XyzReferenceWhite[0] * finv((labVal[0] + .16f) / 1.16f + labVal[1] / 5.00f);
		float Y = XyzReferenceWhite[1] * finv((labVal[0] + .16f) / 1.16f);
		float Z = XyzReferenceWhite[2] * finv((labVal[0] + .16f) / 1.16f - labVal[2] / 2.00f);

		float R = XyzToRgbMatrix[0][0] * X + XyzToRgbMatrix[0][1] * Y + XyzToRgbMatrix[0][2] * Z;
		float G = XyzToRgbMatrix[1][0] * X + XyzToRgbMatrix[1][1] * Y + XyzToRgbMatrix[1][2] * Z;
		float B = XyzToRgbMatrix[2][0] * X + XyzToRgbMatrix[2][1] * Y + XyzToRgbMatrix[2][2] * Z;

		rgb.color[0] = R;
		rgb.color[1] = G;
		rgb.color[2] = B;

		return rgb;
	}

	// compute distance of 2 colors via euclidian vector norm
	__host__ __device__ static float ColorDistance(float* a, float* b, int n = 3)
	{
		float distance = 0;

		for(int i = 0; i < n; ++i)
		{
			float d = a[i] - b[i];
			distance += d * d;
		}

		distance = sqrtf(distance);

		return distance;
	}

	// returns a color given as rgb int [0, 255] into rgb float [0,1]
	__host__ static float* MakeColor(int r, int g, int b)
	{
		float* c = (float*) malloc(sizeof(float) * 3);
		c[0] = r / 255.0f;
		c[1] = g / 255.0f;
		c[2] = b / 255.0f;

		return c;
	}

private:

	//functions used in color conversion

	__host__ __device__ static float f(float v)
	{
		float delta = 6.0f / 29.0f;

		if (v > delta * delta * delta)
		{
			return cbrtf(v);
		}
		else
		{
			return (v) / (3.0f * delta * delta) + 4.0f / 29.0f;
		}
	}

	__host__ __device__ static float finv(float v)
	{
		float delta = 6.0f / 29.0f;

		if (v > delta)
		{
			return v * v * v;
		}
		else
		{
			return 3.0f * delta * delta * delta * (v - (4.0f / 29.0f));
		}
	}
};
