#pragma once
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
		Color lab;

		float X = RgbToXyzMatrix[0][0] * rgbVal[0] + RgbToXyzMatrix[0][1] * rgbVal[1] + RgbToXyzMatrix[0][2] * rgbVal[2];
		float Y = RgbToXyzMatrix[1][0] * rgbVal[0] + RgbToXyzMatrix[1][1] * rgbVal[1] + RgbToXyzMatrix[1][2] * rgbVal[2];
		float Z = RgbToXyzMatrix[2][0] * rgbVal[0] + RgbToXyzMatrix[2][1] * rgbVal[1] + RgbToXyzMatrix[2][2] * rgbVal[2];

		float L = 1.16 * f(Y / XyzReferenceWhite[1]) - .16;
		float a = 5.00 * (f(X / XyzReferenceWhite[0]) - f(Y / XyzReferenceWhite[1]));
		float b = 2.00 * (f(Y / XyzReferenceWhite[1]) - f(Z / XyzReferenceWhite[2]));

		lab.color[0] = L;
		lab.color[1] = a;
		lab.color[2] = b;

		return lab;
	}

	//Converts a color in Lab color space (values [0.0, 1.0]) to the sRGB (values [0.0, 1.0])
	__host__ __device__ static Color LabToRgb(float* labVal)
	{
		Color rgb;

		float X = XyzReferenceWhite[0] * finv((labVal[0] + .16) / 1.16 + labVal[1] / 5.00);
		float Y = XyzReferenceWhite[1] * finv((labVal[0] + .16) / 1.16);
		float Z = XyzReferenceWhite[2] * finv((labVal[0] + .16) / 1.16 - labVal[2] / 2.00);

		float R = XyzToRgbMatrix[0][0] * X + XyzToRgbMatrix[0][1] * Y + XyzToRgbMatrix[0][2] * Z;
		float G = XyzToRgbMatrix[1][0] * X + XyzToRgbMatrix[1][1] * Y + XyzToRgbMatrix[1][2] * Z;
		float B = XyzToRgbMatrix[2][0] * X + XyzToRgbMatrix[2][1] * Y + XyzToRgbMatrix[2][2] * Z;

		rgb.color[0] = R;
		rgb.color[1] = G;
		rgb.color[2] = B;

		return rgb;
	}

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

private:
	static const float RgbToXyzMatrix[3][3];
	static const float XyzToRgbMatrix[3][3];

	static const float XyzReferenceWhite[3];

	static const float delta;

	__host__ __device__ static float f(float v)
	{
		if (v > delta * delta * delta)
		{
			return cbrtf(v);
		}
		else
		{
			return (v) / (3 * delta * delta) + 4. / 29.;
		}
	}

	__host__ __device__ static float finv(float v)
	{
		if (v > delta)
		{
			return v * v * v;
		}
		else
		{
			return 3 * delta * delta * delta * (v - (4. / 29.));
		}
	}
};
