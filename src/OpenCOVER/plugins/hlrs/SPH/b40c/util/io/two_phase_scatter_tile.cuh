/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Kernel utilities for two-phase tile scattering
 ******************************************************************************/

#pragma once

#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

namespace b40c {
namespace util {
namespace io {


/**
 * Performs a two-phase tile scattering in order to improve global-store write
 * coalescing: first to smem, then to global.
 *
 * Does not barrier after usage: a subsequent sync is needed to make shared memory
 * coherent for re-use
 */
template <
	int LOG_LOADS_PER_TILE, 									// Number of vector loads (log)
	int LOG_LOAD_VEC_SIZE,										// Number of items per vector load (log)
	int ACTIVE_THREADS,
	st::CacheModifier CACHE_MODIFIER,
	bool CHECK_ALIGNMENT>
struct TwoPhaseScatterTile
{
	enum {
		LOG_ELEMENTS_PER_THREAD		= LOG_LOADS_PER_TILE + LOG_LOAD_VEC_SIZE,
		ELEMENTS_PER_THREAD			= 1 << LOG_ELEMENTS_PER_THREAD,
		LOADS_PER_TILE 				= 1 << LOG_LOADS_PER_TILE,
		LOAD_VEC_SIZE 				= 1 << LOG_LOAD_VEC_SIZE,
		TILE_SIZE					= ELEMENTS_PER_THREAD * ACTIVE_THREADS,
	};

	template <
		typename T,
		typename Flag,
		typename Rank,
		typename SizeT>
	__device__ __forceinline__ void Scatter(
		T data[LOADS_PER_TILE][LOAD_VEC_SIZE],								// Elements of data to scatter
		Flag flags[LOADS_PER_TILE][LOAD_VEC_SIZE],							// Valid predicates for data elements
		Rank ranks[LOADS_PER_TILE][LOAD_VEC_SIZE],							// Local ranks of data to scatter
		Rank valid_elements,												// Number of valid elements
		T smem_exchange[TILE_SIZE],											// Smem swap exchange storage
		T *d_out,															// Global output to scatter to
		SizeT cta_offset)													// CTA offset into d_out at which to scatter to
	{
		// Scatter valid data into smem exchange, predicated on head_flags
		ScatterTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			ACTIVE_THREADS,
			st::NONE>::Scatter(
				smem_exchange,
				data,
				flags,
				ranks);

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		// Tile of compacted elements
		T compacted_data[ELEMENTS_PER_THREAD][1];

		// Gather compacted data from smem exchange (in 1-element stride loads)
		LoadTile<
			LOG_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			ACTIVE_THREADS,
			ld::NONE,
			false>::LoadValid(							// No need to check alignment
				compacted_data,
				smem_exchange,
				0,
				valid_elements);

		// Scatter compacted data to global output
		util::io::StoreTile<
			LOG_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			ACTIVE_THREADS,
			CACHE_MODIFIER,
			CHECK_ALIGNMENT>::Store(
				compacted_data,
				d_out,
				cta_offset,
				valid_elements);
	}
};



} // namespace io
} // namespace util
} // namespace b40c

