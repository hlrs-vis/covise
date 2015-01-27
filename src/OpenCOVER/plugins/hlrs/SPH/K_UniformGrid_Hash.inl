// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef __UniformGrid_cu__
#define __UniformGrid_cu__

#include "K_UniformGrid_Utils.inl"

// Calculate a grid hash value for each particle

__global__ void K_Grid_Hash (
							   uint				numParticles,
							   float_vec*		dParticlePositions,	
							   GridData			dGridData
							   )
{			
	// particle index
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	// particle position
	float4 p = dParticlePositions[index];

	// get address in grid
	int3 gridPos = UniformGridUtils::calcGridCell(make_float3(p), cGridParams.grid_min, cGridParams.grid_delta);
	uint hash = UniformGridUtils::calcGridHash<true>(gridPos, cGridParams.grid_res);

	// store grid hash and particle index
	dGridData.sort_hashes[index] = hash;
	dGridData.sort_indexes[index] = index;

}


#endif
