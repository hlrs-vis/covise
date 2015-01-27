#ifndef __K_UniformGrid_Utils_cu__
#define __K_UniformGrid_Utils_cu__

#include "K_Common.inl"

namespace UniformGridUtils
{
	// find the grid cell from a position in world space
	static __device__ int3 calcGridCell(float3 const &p, float3 grid_min, float3 grid_delta)
	{
		// subtract grid_min (cell position) and multiply by delta
		return make_int3((p-grid_min) * grid_delta);
	}


	// calculate hash from grid cell
	template <bool wrapEdges>
	static __device__ uint calcGridHash(int3 const &gridPos, float3 grid_res)
	{
		int gx,gy,gz;
		if(wrapEdges)
		{
			int gsx = (int)floor(grid_res.x);
			int gsy = (int)floor(grid_res.y);
			int gsz = (int)floor(grid_res.z);

// 			//power of 2 wrapping..
// 			gx = gridPos.x & gsx-1;
// 			gy = gridPos.y & gsy-1;
// 			gz = gridPos.z & gsz-1;

			// wrap grid... but since we can not assume size is power of 2 we can't use binary AND/& :/
			gx = gridPos.x % gsx;
			gy = gridPos.y % gsy;
			gz = gridPos.z % gsz;
			if(gx < 0) gx+=gsx;
			if(gy < 0) gy+=gsy;
			if(gz < 0) gz+=gsz;
		}
		else
		{
			gx = gridPos.x;
			gy = gridPos.y;
			gz = gridPos.z;
		}

		//return  __mul24(__mul24(gz, (int) cGridParams.grid_res.y)+gy, (int) cGridParams.grid_res.x) + gx;

		//We choose to simply traverse the grid cells along the x, y, and z axes, in that order. The inverse of
		//this space filling curve is then simply:
		// index = x + y*width + z*width*height
		//This means that we process the grid structure in "depth slice" order, and
		//each such slice is processed in row-column order.
		return __mul24(__umul24(gz, grid_res.y), grid_res.x) + __mul24(gy, grid_res.x) + gx;
	}


	// Iterate over particles found in the nearby cells (including cell of position_i)
	template<class O, class D>
	static __device__ void IterateParticlesInCell(
		D 					&data,
		int3 const			&cellPos,
		uint const			&index_i,
		float3 const		&position_i,
		GridData const		&dGridData
)
	{
		// get hash (of position) of current cell
		volatile uint cellHash = UniformGridUtils::calcGridHash<true>(cellPos, cGridParams.grid_res);

		// get start/end positions for this cell/bucket
		//uint startIndex	= FETCH_NOTEX(dGridData,cell_indexes_start,cellHash);
		volatile uint startIndex = FETCH(dGridData,cell_indexes_start,cellHash);

		// check cell is not empty
		if (startIndex != 0xffffffff)
		{
			//uint endIndex = FETCH_NOTEX(dGridData,cell_indexes_end,cellHash);
			volatile uint endIndex = FETCH(dGridData, cell_indexes_end, cellHash);

			// iterate over particles in this cell
			for(uint index_j=startIndex; index_j < endIndex; index_j++)
			{
				O::ForPossibleNeighbor(data, index_i, index_j, position_i);
			}
		}
	}

	// Iterate over particles found in the nearby cells (including cell of position_i)
	template<class O, class D>
	static __device__ void IterateParticlesInNearbyCells(
		D 					&data,
		uint const			&index_i,
		float3 const		&position_i,
		GridData const		&dGridData)
	{
		O::PreCalc(data, index_i);

		// get cell in grid for the given position
		volatile int3 cell = UniformGridUtils::calcGridCell(position_i, cGridParams.grid_min, cGridParams.grid_delta);

		// iterate through the 3^3 cells in and around the given position
		// can't unroll these loops, they are not innermost
		for(int z=cell.z-1; z<=cell.z+1; ++z)
		{
			for(int y=cell.y-1; y<=cell.y+1; ++y)
			{
				for(int x=cell.x-1; x<=cell.x+1; ++x)
				{
					IterateParticlesInCell<O,D>(data, make_int3(x,y,z), index_i, position_i, dGridData);
				}
			}
		}

		O::PostCalc(data, index_i);
	}
	// Iterate over particles found in the neighbor list
	template<class O, class D>
	static __device__ void IterateParticlesInNearbyCells(
		D 					&data,
		uint const			&index_i,
		float3 const		&position_i,
		NeighborList const	&dNeighborList
		)
	{
		O::PreCalc(data, index_i);

		// iterate over particles in neighbor list
		for(uint counter=0; counter < dNeighborList.MAX_NEIGHBORS; counter++)
		{
			//const uint index_j = FETCH(dNeighborList,neighbors, index_i*dNeighborList.neighbors_pitch+counter);
			const uint index_j = FETCH_NOTEX(dNeighborList,neighbors, index_i*dNeighborList.MAX_NEIGHBORS+counter);

			// no more neighbors for this particle
			if(index_j == 0xffffffff)
				break;

			O::ForPossibleNeighbor(data, index_i, index_j, position_i);

		}

		O::PostCalc(data, index_i);
	}

};

#endif
