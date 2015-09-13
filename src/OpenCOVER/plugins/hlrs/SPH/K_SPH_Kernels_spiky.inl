#ifndef __K_SPH_Kernels_Wspiky_cu__
#define __K_SPH_Kernels_Wspiky_cu__

// Spiky kernel by Desbrun and Gascuel, also used by Müller et al.
class Wspiky
{
public:

	static __device__ __host__ float Kernel_Constant(float smoothing_length)
	{
		return 15.0f / ((float)M_PI * pow(smoothing_length, 6.0f) );
	}

	static __device__ __host__ float Kernel_Variable(float smoothing_length, float3 r, float rlen)
	{
		// h - |r|
		float h_rlen =  (smoothing_length - rlen);
		return h_rlen*h_rlen*h_rlen;
	}

	static __device__ __host__ float3 Gradient(float smoothing_length, float3 r, float rlen)
	{
		return Gradient_Constant(smoothing_length) * Gradient_Variable(smoothing_length, r, rlen);
	}

	static __device__ __host__ float Gradient_Constant(float smoothing_length)
	{
		return -45.0f / ((float)M_PI * pow(smoothing_length, 6.0f) );
	}

	static __device__ __host__ float3 Gradient_Variable(float smoothing_length, float3 r, float rlen)
	{
		// h - |r|
		float h_rlen = (smoothing_length-rlen);
		return r*(1.0f/rlen)*(h_rlen*h_rlen);
	}

	static __device__ __host__ float Laplace_Constant(float smoothing_length)
	{
		return -90.0f / ((float)M_PI * pow(smoothing_length, 6.0f) );
	}

	static __device__ __host__ float3 Laplace_Variable(float smoothing_length, float3 r, float rlen)
	{
		// h - |r|
		float h_rlen = (smoothing_length-rlen);
		float h_2rlen = (smoothing_length-2*rlen);
		return (1.0f/r) * (h_rlen*h_2rlen);
	}
};

#endif