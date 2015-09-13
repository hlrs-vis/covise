#ifndef __K_SPH_Kernels_quadratic_cu__
#define __K_SPH_Kernels_quadratic_cu__

// from "PhD Thesis: Application of the Smoothed Particle Hydrodynamics model SPHysics to free-surface hydrodynamics
class Wquadratic
{
public:

	static __device__ __host__ float Kernel_Constant(float smoothing_length)
	{
		// for 2d
		//float c = 2/(M_PI * smoothing_length * smoothing_length);
		// for 3d
		float c = 5.0f/(4.0f*(float)M_PI*smoothing_length*smoothing_length*smoothing_length);
		
		return c;
	}

	static __device__ __host__ float Kernel_Variable(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen)
	{
		float q = rlen/smoothing_length;
		
		if(0<=q && q<=2)
		{
			return 0.1875f*q*q - 0.75f*q + 0.75f;
		}
		return 0.f;
	}	

	static __device__ __host__ float Gradient_Constant(float smoothing_length)
	{
		//TODO
		return 0.0f;
	}

	static __device__ __host__ float Gradient_Variable(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen)
	{
		return 0.0f;
	}

};

#endif
