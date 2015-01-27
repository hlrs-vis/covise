#ifndef COMMON_CUH
#define COMMON_CUH

#include "Config.h"

#ifdef SPHSIMLIB_VEC_TYPE_FLOAT4
typedef float4 float_vec;
#else
typedef float3 float_vec;
#endif

struct matrix3
{
	// a float_vec for each row.
	float_vec r1;
	float_vec r2;
	float_vec r3;
};

#endif //COMMON_CUH