#ifndef __K_SPH_Common_cu__
#define __K_SPH_Common_cu__

template<class O, class D>
class SPHNeighborCalc
{
public:
	// this is called before the loop over each neighbor particle
	static __device__ void PreCalc(D &data, uint index_i)
	{
		O::PreCalc(data, index_i);
	}

	static __device__ void ForNeighbor(D &data, uint const &index_i, uint const &index_j, float3 const &r, float const &rlen)
	{
		O::ForNeighbor(data, index_i, index_j, r, rlen);
	}		

	// this is called after the loop over each particle in a cell
	static __device__ void PostCalc(D &data, uint index_i)
	{
		O::PostCalc(data, index_i);
	}

	// this is called inside the loop over each particle in a cell
	static __device__ void ForPossibleNeighbor(D &data, uint const &index_i, uint const &index_j, float3 const &position_i)
	{
		// check not colliding with self
		if (index_j != index_i) 
		{		
			// get the particle info (in the current grid) to test against
			float3 position_j = make_float3(FETCH(data.dParticleDataSorted, position, index_j));

			// get the relative distance between the two particles, translate to simulation space
			float3 r = (position_i - position_j) * cFluidParams.scale_to_simulation;

			float rlen_sq = dot(r,r);
			// |r|
			float rlen = sqrtf(rlen_sq);

			// is this particle within cutoff?
			if (rlen <= cFluidParams.smoothing_length) 
			{
				O::ForNeighbor(data, index_i, index_j, r, rlen, rlen_sq);
			}
		}
	}

};

#endif