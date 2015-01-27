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
 * Cooperative tile SOA (structure-of-arrays) reduction within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/soa/serial_soa_reduce.cuh>
#include <b40c/util/reduction/soa/warp_soa_reduce.cuh>
#include <b40c/util/scan/soa/warp_soa_scan.cuh>

namespace b40c {
namespace util {
namespace reduction {
namespace soa {


/**
 * Cooperative SOA reduction in SRTS smem grid hierarchies
 */
template <
	typename SrtsSoaDetails,
	typename SecondarySrtsSoaDetails = typename SrtsSoaDetails::SecondarySrtsSoaDetails>
struct CooperativeSoaGridReduction;


/**
 * Cooperative SOA tile reduction
 */
template <int VEC_SIZE>
struct CooperativeSoaTileReduction
{
	//---------------------------------------------------------------------
	// Iteration structures for reducing tile SOA vectors into their
	// corresponding SRTS lanes
	//---------------------------------------------------------------------

	// Next lane/load
	template <int LANE, int TOTAL_LANES>
	struct ReduceLane
	{
		template <
			typename SrtsSoaDetails,
			typename TileSoa,
			typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			SrtsSoaDetails srts_soa_details,
			TileSoa tile_soa,
			ReductionOp reduction_op)
		{
			// Reduce the partials in this lane/load
			typename SrtsSoaDetails::TileTuple partial_reduction;
			SerialSoaReduce<VEC_SIZE>::Reduce(
				partial_reduction, tile_soa, LANE, reduction_op);

			// Store partial reduction into SRTS grid
			srts_soa_details.lane_partials.Set(partial_reduction, LANE, 0);

			// Next load
			ReduceLane<LANE + 1, TOTAL_LANES>::Invoke(
				srts_soa_details, tile_soa, reduction_op);
		}
	};

	// Terminate
	template <int TOTAL_LANES>
	struct ReduceLane<TOTAL_LANES, TOTAL_LANES>
	{
		template <
			typename SrtsSoaDetails,
			typename TileSoa,
			typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			SrtsSoaDetails srts_soa_details,
			TileSoa tile_soa,
			ReductionOp reduction_op) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Reduce a single tile.  Carry is computed (or updated if REDUCE_INTO_CARRY is set)
	 * only in last raking thread
	 *
	 * Caution: Post-synchronization is needed before grid reuse.
	 */
	template <
		bool REDUCE_INTO_CARRY,
		typename SrtsSoaDetails,
		typename TileSoa,
		typename TileTuple,
		typename ReductionOp>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsSoaDetails srts_soa_details,
		TileSoa tile_soa,
		TileTuple &carry,
		ReductionOp reduction_op)
	{
		// Reduce vectors in tile, placing resulting partial into corresponding SRTS grid lanes
		ReduceLane<0, SrtsSoaDetails::SCAN_LANES>::Invoke(
			srts_soa_details, tile_soa, reduction_op);

		__syncthreads();

		CooperativeSoaGridReduction<SrtsSoaDetails>::template ReduceTileWithCarry<REDUCE_INTO_CARRY>(
			srts_soa_details, carry, reduction_op);
	}

	/**
	 * Reduce a single tile.  Result is computed and returned in all threads.
	 *
	 * No post-synchronization needed before srts_details reuse.
	 */
	template <
		typename TileTuple,
		typename SrtsSoaDetails,
		typename TileSoa,
		typename ReductionOp>
	static __device__ __forceinline__ void ReduceTile(
		TileTuple &retval,
		SrtsSoaDetails srts_soa_details,
		TileSoa tile_soa,
		ReductionOp reduction_op)
	{
		// Reduce vectors in tile, placing resulting partial into corresponding SRTS grid lanes
		ReduceLane<0, SrtsSoaDetails::SCAN_LANES>::Invoke(
			srts_soa_details, tile_soa, reduction_op);

		__syncthreads();

		return CooperativeSoaGridReduction<SrtsSoaDetails>::ReduceTile(
			srts_soa_details, reduction_op);
	}
};




/******************************************************************************
 * CooperativeSoaGridReduction
 ******************************************************************************/

/**
 * Cooperative SOA SRTS grid reduction (specialized for last-level of SRTS grid)
 */
template <typename SrtsSoaDetails>
struct CooperativeSoaGridReduction<SrtsSoaDetails, NullType>
{
	typedef typename SrtsSoaDetails::TileTuple TileTuple;

	/**
	 * Reduction in last-level SRTS grid.  Carry is assigned (or reduced into
	 * if REDUCE_INTO_CARRY is set), but only in last raking thread
	 */
	template <
		bool REDUCE_INTO_CARRY,
		typename ReductionOp>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsSoaDetails srts_soa_details,
		TileTuple &carry,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			TileTuple inclusive_partial;
			SerialSoaReduce<SrtsSoaDetails::PARTIALS_PER_SEG>::Reduce(
				inclusive_partial,
				srts_soa_details.raking_segments,
				reduction_op);

			// Inclusive warp scan that sets warpscan total in all
			// raking threads. (Use warp scan instead of warp reduction
			// because the latter supports non-commutative reduction
			// operators)
			TileTuple warpscan_total;
			scan::soa::WarpSoaScan<
				SrtsSoaDetails::LOG_RAKING_THREADS,
				false>::Scan(
					inclusive_partial,
					warpscan_total,
					srts_soa_details.warpscan_partials,
					reduction_op);

			// Update/set carry
			carry = (REDUCE_INTO_CARRY) ?
				reduction_op(carry, warpscan_total) :
				warpscan_total;
		}
	}


	/**
	 * Reduction in last-level SRTS grid.  Result is computed in all threads.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ TileTuple ReduceTile(
		SrtsSoaDetails srts_soa_details,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			TileTuple inclusive_partial = SerialSoaReduce<SrtsSoaDetails::PARTIALS_PER_SEG>::Reduce(
				srts_soa_details.raking_segments, reduction_op);

			// Warp reduction
			TileTuple warpscan_total = WarpSoaReduce<SrtsSoaDetails::LOG_RAKING_THREADS>::ReduceInLast(
				inclusive_partial,
				srts_soa_details.warpscan_partials,
				reduction_op);
		}

		__syncthreads();

		// Return last thread's inclusive partial
		return srts_soa_details.CumulativePartial();
	}
};


} // namespace soa
} // namespace reduction
} // namespace util
} // namespace b40c

