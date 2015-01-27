#ifndef __K_SPH_Kernels_quintic_cu__
#define __K_SPH_Kernels_quintic_cu__

// TODO 
// see crespo_thesis.pdf for summary of kernels and tensile correction terms etc!

// the quintic wendland kernel [Wendland, 1995]
class Wquintic
{
public:

	static __device__ __host__ float Kernel(float smoothing_length, float3 r, float rlen, float rlen_sq)
	{
		float Q = rlen / smoothing_length;
		if(Q < 2)
		{
			// for 2D
			//float c = 7.0f/(4.0f*M_PI*rlen_sq);
			// for 3D
			float c = 7.0f/(8.0f*M_PI*rlen*rlen_sq);
			return c * pow(1-0.5f*Q, 4) * (2*Q+1);
		}
		return 0;
	}

	static __device__ __host__ float3 Gradient(float smoothing_length, float3 r, float rlen, float rlen_sq)
	{
		float Q = rlen / smoothing_length;
		if(Q < 2)
		{
			// for 2D
			//scalar c = (-35 * M_1_PI / 4 * rlen_sq*rlen_sq);
			// for 3D
			float c =  (-35 * M_1_PI) / (8 * rlen_sq*rlen_sq);
			float dif = 2 - Q;
			return r * (c * dif * dif / r);
		}
		return make_float3(0,0,0);
	}


};
#endif
