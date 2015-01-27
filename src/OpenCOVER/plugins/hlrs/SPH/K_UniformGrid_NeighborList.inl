#ifndef __K_UniformGrid_NeighborList_cu__
#define __K_UniformGrid_NeighborList_cu__

class UniformGridNeighborList
{
public:

	struct Data
	{
		ParticleData dParticlesSorted;
		NeighborList dNeighborList;

		uint neighbor_counter;		
	};


	class Calc
	{
	public:

		static __device__ void PreCalc(Data &data, uint const &index_i)
		{
			data.neighbor_counter = 0;
		}

		static __device__ void ForPossibleNeighbor(Data &data, uint const &index_i, uint const &index_j, float3 const &position_i)
		{
			// check not colliding with self
			if (data.neighbor_counter < data.dNeighborList.MAX_NEIGHBORS && index_j != index_i) 
			{  
				// get the particle position (in the current cell) to test against
				float3 position_j = make_float3(FETCH(data.dParticlesSorted, position, index_j));

				// get the relative distance between the two particles, translate to simulation space
				float3 r = (position_i - position_j) * cFluidParams.scale_to_simulation;

				float rlen_sq = dot(r,r);
				// |r|
				float rlen = sqrtf(rlen_sq);

				// is this particle within cutoff?
				if (rlen < cFluidParams.smoothing_length) 
				{	
					ForNeighbor(data, index_i, index_j, r, rlen, rlen_sq);
				}
			}
		}

		static __device__ void ForNeighbor(Data &data, uint const &index_i, uint const &index_j, float3 const &r, float const& rlen, float const &rlen_sq)
		{
			data.dNeighborList.neighbors[index_i*data.dNeighborList.neighbors_pitch + data.neighbor_counter] = index_j;
			data.neighbor_counter++;

			// 					if(blockIdx.x == 0 && threadIdx.x == 0)
			// 						cuPrintf("ORG: %d %f\n", index_j, rlen);
		}

		static __device__ void PostCalc(Data &data, uint index_i)
		{
		}

	};
};

__global__ void buildNeighborList (
	uint					numParticles,
	NeighborList			dNeighborList,
	ParticleData			dParticlesSorted,
	GridData				dGridData
	)
{
	// particle index	
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;		
	if (index >= numParticles) return;

	UniformGridNeighborList::Data data;
	data.dParticlesSorted = dParticlesSorted;
	data.dNeighborList = dNeighborList;

	// read particle data from sorted arrays
	float3 position_i = make_float3(FETCH(dParticlesSorted, position, index));

	// Do calculations on particles in neighboring cells
	UniformGridUtils::IterateParticlesInNearbyCells
		<UniformGridNeighborList::Calc, UniformGridNeighborList::Data>
		(data, index, position_i, dGridData);
}

#endif