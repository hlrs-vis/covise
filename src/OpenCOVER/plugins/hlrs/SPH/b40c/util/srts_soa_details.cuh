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
 * Operational details for threads working in an SOA (structure of arrays)
 * SRTS grid
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>

namespace b40c {
namespace util {


/**
 * Operational details for threads working in an SRTS grid
 */
template <
	typename TileTuple,
	typename SrtsGridTuple,
	int Grids = SrtsGridTuple::NUM_FIELDS>
struct SrtsSoaDetails;


/**
 * Two-field SRTS details
 */
template <
	typename _TileTuple,
	typename SrtsGridTuple>
struct SrtsSoaDetails<_TileTuple, SrtsGridTuple, 2> : SrtsGridTuple::T0
{
	enum {
		CUMULATIVE_THREAD 	= SrtsSoaDetails::RAKING_THREADS - 1,
		WARP_THREADS 		= B40C_WARP_THREADS(SrtsSoaDetails::CUDA_ARCH)
	};

	// Simple SOA tuple "slice" type
	typedef _TileTuple TileTuple;

	// SOA type of raking lanes
	typedef Tuple<
		typename TileTuple::T0*,
		typename TileTuple::T1*> GridStorageSoa;

	// SOA type of warpscan storage
	typedef Tuple<
		typename SrtsGridTuple::T0::WarpscanT (*)[WARP_THREADS],
		typename SrtsGridTuple::T1::WarpscanT (*)[WARP_THREADS]> WarpscanSoa;

	// SOA type of partial-insertion pointers
	typedef Tuple<
		typename SrtsGridTuple::T0::LanePartial,
		typename SrtsGridTuple::T1::LanePartial> LaneSoa;

	// SOA type of raking segments
	typedef Tuple<
		typename SrtsGridTuple::T0::RakingSegment,
		typename SrtsGridTuple::T1::RakingSegment> RakingSoa;

	// SOA type of secondary grids
	typedef Tuple<
		typename SrtsGridTuple::T0::SecondaryGrid,
		typename SrtsGridTuple::T1::SecondaryGrid> SecondarySrtsGridTuple;

	typedef typename If<Equals<NullType, typename SecondarySrtsGridTuple::T0>::VALUE,
		NullType,
		SrtsSoaDetails<TileTuple, SecondarySrtsGridTuple> >::Type SecondarySrtsSoaDetails;

	/**
	 * Warpscan storages
	 */
	WarpscanSoa warpscan_partials;

	/**
	 * Lane insertion/extraction pointers.
	 */
	LaneSoa lane_partials;

	/**
	 * Raking pointers
	 */
	RakingSoa raking_segments;


	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ SrtsSoaDetails(
		GridStorageSoa smem_pools,
		WarpscanSoa warpscan_partials) :

			warpscan_partials(warpscan_partials),
			lane_partials(												// set lane partial pointer
				SrtsGridTuple::T0::MyLanePartial(smem_pools.t0),
				SrtsGridTuple::T1::MyLanePartial(smem_pools.t1))
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Set raking segment pointers
			raking_segments = RakingSoa(
				SrtsGridTuple::T0::MyRakingSegment(smem_pools.t0),
				SrtsGridTuple::T1::MyRakingSegment(smem_pools.t1));
		}
	}


	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ SrtsSoaDetails(
		GridStorageSoa smem_pools,
		WarpscanSoa warpscan_partials,
		TileTuple soa_tuple_identity) :

			warpscan_partials(warpscan_partials),
			lane_partials(												// set lane partial pointer
				SrtsGridTuple::T0::MyLanePartial(smem_pools.t0),
				SrtsGridTuple::T1::MyLanePartial(smem_pools.t1))
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Set raking segment pointers
			raking_segments = RakingSoa(
				SrtsGridTuple::T0::MyRakingSegment(smem_pools.t0),
				SrtsGridTuple::T1::MyRakingSegment(smem_pools.t1));

			// Initialize first half of warpscan storages to identity
			warpscan_partials.Set(soa_tuple_identity, 0, threadIdx.x);
		}
	}


	/**
	 * Return the cumulative partial left in the final warpscan cell
	 */
	__device__ __forceinline__ TileTuple CumulativePartial()
	{
		TileTuple retval;
		warpscan_partials.Get(retval, 1, CUMULATIVE_THREAD);
		return retval;
	}
};








} // namespace util
} // namespace b40c

