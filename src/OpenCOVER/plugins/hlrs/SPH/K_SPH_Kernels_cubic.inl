#ifndef __K_SPH_Kernels_cubic_cu__
#define __K_SPH_Kernels_cubic_cu__

// TODO 
// see crespo_thesis.pdf for summary of kernels and tensile correction terms etc!
// add tensile correction terms!

// used by "A fully explicit three-step SPH algorithm for simulation of non-Newtonian fluid flow"

//third order B-spline
class Wcubic
{
public:

	static __device__ __host__ float Kernel(float smoothing_length, float3 r, float rlen)
	{
		float Q = rlen / smoothing_length;

		if(Q <= 1)
		{
			// for 2D
			//float c = 10 * M_1_PI / 7 * smoothInvSq;
			// for 3D
			float c = 1.0f/((float)M_PI*(smoothing_length*smoothing_length*smoothing_length));
			return c * (1 - 1.5f*Q*Q + 0.75f*Q*Q*Q);
		}
		else if(Q <= 2)
		{
			// for 2D
			//float c = 10 * M_1_PI / 28 * smoothInvSq;
			// for 3D
			float c = 0.25f/((float)M_PI/(smoothing_length*smoothing_length*smoothing_length));

			float dif = Q-2;
			return - c * dif * dif * dif;
		}
		return 0;
	}

	static __device__ __host__ float3 Gradient(float smoothing_length, float smoothing_length_pow2, float smoothing_length_pow3, float smoothing_length_pow4, float3 r, float rlen, float rlen_sq)
	{
		float Q = rlen / smoothing_length;

		if(Q <= 1)
		{
			// for 3D
			float c =  1 / ((float)M_PI *(smoothing_length_pow3));
			return - r * c * ( 3/(smoothing_length_pow2) - (9*rlen)/(4*smoothing_length_pow3) );
		}
		else if(Q <= 2)
		{
			// for 3D
			float c = 3 / ( 4* (float)M_PI * (smoothing_length_pow4));
			float dif = Q-2;
			return - r * (c * dif * dif) / rlen;
		}
		return make_float3(0.0f);
	}

};
#endif
