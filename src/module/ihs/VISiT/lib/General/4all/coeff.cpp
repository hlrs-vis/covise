#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <windows.h>
#else 
#include <strings.h>
#endif
#include <math.h>

void DetermineCoefficients(float *x, float *y, float *a)
{
	float buf;

	a[2]  = (y[1] - y[2]) * (x[0] - x[1]) - (y[0] - y[1]) * (x[1] - x[2]);
	buf	  = (pow(x[1], 2) - pow(x[2], 2)) * (x[0] - x[1]);
	buf	 -= ((pow(x[0], 2) - pow(x[1], 2)) * (x[1] - x[2]));
	a[2] /= buf;
	a[1]  = ((y[0] - y[1]) - (pow(x[0], 2) - pow(x[1], 2)) * a[2]) / (x[0] - x[1]);
	a[0]  = y[2] - pow(x[2], 2) * a[2] - x[2] * a[1];
}


float EvaluateParameter(float x, float *a)
{
	float val;

	val = a[2] * pow(x, 2) + a[1] * x + a[0];
	return val;
}
