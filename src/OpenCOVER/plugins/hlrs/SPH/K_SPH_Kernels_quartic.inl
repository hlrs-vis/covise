#ifndef __K_SPH_Kernels_quartic_cu__
#define __K_SPH_Kernels_quartic_cu__

//from liu et al, "Constructing smoothing functions in smoothed particle hydrodynamics with applications"
class Wquartic
{
public:

	static __device__ __host__ float Kernel(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen)
	{
		//TODO
		return 0.0f;
	}	

	static __device__ __host__ float Gradient(float smoothing_length)
	{
		//TODO
		return 0.0f;
	}


};

#endif
