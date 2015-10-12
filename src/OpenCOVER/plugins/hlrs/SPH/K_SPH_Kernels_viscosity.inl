#ifndef __K_SPH_Kernels_Wviscosity_cu__
#define __K_SPH_Kernels_Wviscosity_cu__

// Viscosity kernel from Müller et al.
class Wviscosity
{
public:

	static __device__ __host__ float Kernel_Constant(float smoothing_length)
	{
		return 15.0f / ((float)M_PI * pow(smoothing_length, 6.0f) );
	}

	static __device__ __host__ float Kernel_Variable(float smoothing_length, float3 r, float rlen)
	{
		float h_rlen =  (smoothing_length - rlen);
		return h_rlen*h_rlen*h_rlen;
	}

	static __device__ __host__ float Gradient_Constant(float smoothing_length)
	{
		return 15.0f / (2 * (float)M_PI * pow(smoothing_length, 3.0f) );
	}

	static __device__ __host__ float Gradient_Variable(float smoothing_length, float3 r, float rlen)
	{
		float part1 = (-3*rlen)/(2*pow(smoothing_length, 3.0f));
		float part2 = (2/smoothing_length*smoothing_length);
		float part3 = -smoothing_length/(2*pow(rlen,3.0f));
		return part1 + part2 + part3;
	}

	static __device__ __host__ float Laplace_Constant(float smoothing_length)
	{
		return 45.0f / ((float)M_PI * pow(smoothing_length, 6.0f) );
	}

	static __device__ __host__ float Laplace_Variable(float smoothing_length, float3 r, float rlen)
	{
		float h_rlen = (smoothing_length-rlen);
		return h_rlen;
	}
};
#endif