#ifndef __UniformGrid_cuh__
#define __UniformGrid_cuh__

#include "Config.h"

#include "SimCudaAllocator.h"
#include "SimCudaHelper.h"

#ifdef SPHSIMLIB_USE_CUDPP_SORT
#include "cudpp.h"
#endif

#ifdef SPHSIMLIB_USE_THRUST_SORT
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#endif

#include "K_Common.cuh"
#include "ParticleData.h"
#include "SimBufferManager.h"
#include "SimBufferCuda.h"
#include "timer.h"

typedef unsigned int uint;

enum UniformGridBuffers
{
	SortHashes,
	SortIndexes,
	CellIndexesStart,
	CellIndexesStop,
};

struct NeighborList
{
	int numParticles;
	int MAX_NEIGHBORS;
	uint* neighbors;
	// pitch, IN ELEMENTS, NOT BYTES
	size_t		neighbors_pitch;
};

struct GridParams
{
	float3			grid_size;
	float3			grid_min;
	float3			grid_max;

	// number of cells in each dimension/side of grid
	float3			grid_res;

	float3			grid_delta;
};

struct GridData
{
	uint* sort_hashes;			// particle hashes
	uint* sort_indexes;			// particle indices
	uint* cell_indexes_start;	// mapping between bucket hash and start index in sorted list
	uint* cell_indexes_end;		// mapping between bucket hash and end index in sorted list
};


class UniformGrid
{
public:
	UniformGrid(SimLib::SimCudaAllocator* SimCudaAllocator,	SimLib::SimCudaHelper *simCudaHelper);
	~UniformGrid();

	void Alloc(uint numParticles, float cellWorldSize, float gridWorldSize);
	void Clear();
	void Free()	;

	float Hash(bool doTiming, float_vec* dParticlePositions, uint numParticles);
	float Sort(bool doTiming);

	GridData GetGridData();
	unsigned int GetNumCells(){return mNumCells;}
	GridParams& GetGridParams(){return dGridParams;}

private:
	void CalculateGridParameters(float cellWorldSize, float gridWorldSize);

	SimLib::BufferManager<UniformGridBuffers> *mGridParticleBuffers;
	SimLib::BufferManager<UniformGridBuffers> *mGridCellBuffers;

	unsigned int mNumParticles;
	unsigned int mNumCells;

	bool mAlloced;

	ocu::GPUTimer *mGPUTimer;

	SimLib::SimCudaAllocator* mSimCudaAllocator;
	SimLib::SimCudaHelper	*mSimCudaHelper;

	GridParams dGridParams;

#ifdef SPHSIMLIB_USE_CUDPP_SORT
	CUDPPHandle m_sortHandle;
#endif

#if (defined SPHSIMLIB_USE_B40C_SORT || defined SPHSIMLIB_USE_CUDA_B40C_SORT)
	void* m_b40c_storage;	
	void* m_b40c_sorting_enactor;
#endif

#ifdef SPHSIMLIB_USE_THRUST_SORT
	thrust::device_ptr<uint>* mThrustKeys;
	thrust::device_ptr<uint>* mThrustVals;
#endif
	int mSortBitsPrecision;
};

#endif
