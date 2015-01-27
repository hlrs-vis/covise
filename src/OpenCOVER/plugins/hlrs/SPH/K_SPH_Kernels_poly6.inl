#ifndef __K_SPH_Kernels_Wpoly6_cu__
#define __K_SPH_Kernels_Wpoly6_cu__

class Wpoly6
{
public:

	static __device__ __host__ float Kernel_Constant(float smoothing_length)
	{
		return 315.0f / (64.0f * M_PI * pow(smoothing_length, 9.0f) );
	}

	static __device__ __host__ float Kernel_Variable(float smoothing_length_pow2, float3 r, float rlen_sq)
	{
		float hsq_rlensq = smoothing_length_pow2 - rlen_sq;
		return hsq_rlensq * hsq_rlensq * hsq_rlensq;	
	}


	static __device__ __host__ float Gradient_Constant(float smoothing_length)
	{
		return -945.0f / (32.0f * M_PI * pow(smoothing_length, 9.0f) );
	}

	static __device__ __host__ float Gradient_Variable(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen)
	{
		// h - |r|^2
		float hsq_rlensq = smoothing_length_pow2 - (rlen*rlen);
		return hsq_rlensq * hsq_rlensq;	
	}

	static __device__ __host__ float Gradient(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen)
	{
		return Gradient_Constant(smoothing_length) * Gradient_Variable(smoothing_length, smoothing_length_pow2, r, rlen);
	}

	static __device__ __host__ float Laplace_Constant(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen)
	{
		return -945.0f / (32.0f * M_PI * pow(smoothing_length, 9.0f) );
	}

	static __device__ __host__ float Laplace_Variable(float smoothing_length, float smoothing_length_pow2, float3 r, float rlen)
	{
		// |r|^2
		float rlen_sq = rlen*rlen;
		// h - |r|^2
		float part1 = smoothing_length_pow2 - rlen_sq;
		// 3h - 7|r|^2
		float part2 = 3.0f*smoothing_length_pow2 - 7.0f*rlen_sq;
		return part1 * part2;	
	}
};

#endif
