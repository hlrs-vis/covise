#ifndef __K_SPH_Kernels_gaussian_cu__
#define __K_SPH_Kernels_gaussian_cu__

// from "PhD Thesis: Application of the Smoothed Particle Hydrodynamics model SPHysics to free-surface hydrodynamics
class Wgaussian
{
public:

	static __device__ __host__ float Kernel_Constant(float smoothing_length, float smoothing_length_pow2)
	{
		// for 2d
		//float c = 1/(M_PI * smoothing_length_pow2);
		// for 3d
		float c = 1/(powf(M_PI, 1.5f)*smoothing_length_pow2*smoothing_length);
		return c;
	}

	static __device__ __host__ float Kernel_Variable(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen, float rlen_sq)
	{
		float Q = rlen/smoothing_length;
		
		if(0<=Q && Q<=2)
		{
			return 1/expf((smoothing_length_pow2*rlen_sq));
		}
		return 0.f;
	}	

	static __device__ __host__ float Kernel(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen, float rlen_sq)
	{
		return Kernel_Constant(smoothing_length,smoothing_length_pow2) * Kernel_Variable(smoothing_length, smoothing_length_pow2, r, rlen, rlen_sq);
	}	

	static __device__ __host__ float Gradient_Constant(float smoothing_length, float smoothing_length_pow2)
	{
		// for 3d
		float c = -2/(powf(M_PI, 0.5f)*smoothing_length_pow2*smoothing_length_pow2*smoothing_length);
		return c;
	}

	static __device__ __host__ float3 Gradient_Variable(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen, float rlen_sq)
	{
		float Q = rlen/smoothing_length;

		if(0<Q && Q<2)
		{
			return r/expf((smoothing_length_pow2*rlen_sq));
		}
		return make_float3(0.f);
	}

	static __device__ __host__ float3 Gradient(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen, float rlen_sq)
	{
		return Gradient_Constant(smoothing_length, smoothing_length_pow2) * Gradient_Variable(smoothing_length, smoothing_length_pow2, r, rlen, rlen_sq);
	}	

};


#endif
