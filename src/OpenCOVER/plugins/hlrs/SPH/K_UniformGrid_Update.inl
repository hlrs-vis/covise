// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef __K_UniformGrid_Update_cu__
#define __K_UniformGrid_Update_cu__

// read/write from the unsorted data structure to the sorted one
template <class T, class D> 
__global__ void K_Grid_UpdateSorted (
								  int		numParticles,
								  D			dParticles,
								  D			dParticlesSorted, 
								  GridData dGridData
								  )
{
	// particle index	
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;		
	if (index >= numParticles) return;

	// blockSize + 1 elements	
	extern __shared__ uint sharedHash[];	

	uint hash = dGridData.sort_hashes[index];

	// Load hash data into shared memory so that we can look 
	// at neighboring particle's hash value without loading
	// two hash values per thread	
	sharedHash[threadIdx.x+1] = hash;
	if (index > 0 && threadIdx.x == 0 ) {

		// first thread in block must load neighbor particle hash
		sharedHash[0] = dGridData.sort_hashes[index-1];
	}

#ifndef __DEVICE_EMULATION__
	__syncthreads ();
#endif

	// If this particle has a different cell index to the previous
	// particle then it must be the first particle in the cell,
	// so store the index of this particle in the cell.
	// As it isn't the first particle, it must also be the cell end of
	// the previous particle's cell

	if ((index == 0 || hash != sharedHash[threadIdx.x]) )
	{
		dGridData.cell_indexes_start[hash] = index;
		if (index > 0)
			dGridData.cell_indexes_end[sharedHash[threadIdx.x]] = index;
	}

	if (index == numParticles - 1)
	{
		dGridData.cell_indexes_end[hash] = index + 1;
	}

	uint sortedIndex = dGridData.sort_indexes[index];

	// Copy data from old unsorted buffer to sorted buffer
	T::UpdateSortedValues(dParticlesSorted, dParticles, index, sortedIndex);
}

#endif