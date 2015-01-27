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
 * Cooperative tile reduction within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/warp_reduce.cuh>

namespace b40c {
namespace util {
namespace reduction {


/**
 * Cooperative reduction in SRTS smem grid hierarchies
 */
template <
	typename SrtsDetails,
	typename SecondarySrtsDetails = typename SrtsDetails::SecondarySrtsDetails>
struct CooperativeGridReduction;


/**
 * Cooperative tile reduction
 */
template <int VEC_SIZE>
struct CooperativeTileReduction
{
	//---------------------------------------------------------------------
	// Iteration structures for reducing tile vectors into their
	// corresponding SRTS lanes
	//---------------------------------------------------------------------

	// Next lane/load
	template <int LANE, int TOTAL_LANES>
	struct ReduceLane
	{
		template <typename SrtsDetails, typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			SrtsDetails srts_details,
			typename SrtsDetails::T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
			ReductionOp reduction_op)
		{
			// Reduce the partials in this lane/load
			typename SrtsDetails::T partial_reduction = SerialReduce<VEC_SIZE>::Invoke(
				data[LANE], reduction_op);

			// Store partial reduction into SRTS grid
			srts_details.lane_partial[LANE][0] = partial_reduction;

			// Next load
			ReduceLane<LANE + 1, TOTAL_LANES>::Invoke(
				srts_details, data, reduction_op);
		}
	};


	// Terminate
	template <int TOTAL_LANES>
	struct ReduceLane<TOTAL_LANES, TOTAL_LANES>
	{
		template <typename SrtsDetails, typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			SrtsDetails srts_details,
			typename SrtsDetails::T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
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
		bool REDUCE_INTO_CARRY, 				// Whether or not to assign carry or reduce into it
		typename SrtsDetails,
		typename ReductionOp>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsDetails srts_details,
		typename SrtsDetails::T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
		typename SrtsDetails::T &carry,
		ReductionOp reduction_op)
	{
		// Reduce partials in each vector-load, placing resulting partials in SRTS smem grid lanes (one lane per load)
		ReduceLane<0, SrtsDetails::SCAN_LANES>::Invoke(srts_details, data, reduction_op);

		__syncthreads();

		CooperativeGridReduction<SrtsDetails>::template ReduceTileWithCarry<REDUCE_INTO_CARRY>(
			srts_details, carry, reduction_op);
	}

	/**
	 * Reduce a single tile.  Result is computed and returned in all threads.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <typename SrtsDetails, typename ReductionOp>
	static __device__ __forceinline__ typename SrtsDetails::T ReduceTile(
		SrtsDetails srts_details,
		typename SrtsDetails::T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
		ReductionOp reduction_op)
	{
		// Reduce partials in each vector-load, placing resulting partials in SRTS smem grid lanes (one lane per load)
		ReduceLane<0, SrtsDetails::SCAN_LANES>::Invoke(srts_details, data, reduction_op);

		__syncthreads();

		return CooperativeGridReduction<SrtsDetails>::ReduceTile(
			srts_details, reduction_op);
	}
};




/******************************************************************************
 * CooperativeGridReduction
 ******************************************************************************/

/**
 * Cooperative SRTS grid reduction (specialized for last-level of SRTS grid)
 */
template <typename SrtsDetails>
struct CooperativeGridReduction<SrtsDetails, NullType>
{
	typedef typename SrtsDetails::T T;

	/**
	 * Reduction in last-level SRTS grid.  Carry is assigned (or reduced into
	 * if REDUCE_INTO_CARRY is set), but only in last raking thread
	 */
	template <
		bool REDUCE_INTO_CARRY,
		typename ReductionOp>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsDetails srts_details,
		T &carry,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, reduction_op);

			// Warp reduction
			T warpscan_total = WarpReduce<SrtsDetails::LOG_RAKING_THREADS>::InvokeSingle(
				partial, srts_details.warpscan, reduction_op);

			carry = (REDUCE_INTO_CARRY) ?
				reduction_op(carry, warpscan_total) : 	// Reduce into carry
				warpscan_total;							// Assign carry
		}
	}


	/**
	 * Reduction in last-level SRTS grid.  Result is computed in all threads.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ T ReduceTile(
		SrtsDetails srts_details,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, reduction_op);

			// Warp reduction
			WarpReduce<SrtsDetails::LOG_RAKING_THREADS>::InvokeSingle(
				partial, srts_details.warpscan, reduction_op);
		}

		__syncthreads();

		// Return last thread's inclusive partial
		return srts_details.CumulativePartial();
	}
};


/**
 * Cooperative SRTS grid reduction for multi-level SRTS grids
 */
template <typename SrtsDetails, typename SecondarySrtsDetails>
struct CooperativeGridReduction
{
	typedef typename SrtsDetails::T T;

	/**
	 * Reduction in SRTS grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	template <bool REDUCE_INTO_CARRY, typename ReductionOp>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsDetails srts_details,
		T &carry,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, reduction_op);

			// Place partial in next grid
			srts_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively reduce in next grid
		CooperativeGridReduction<SecondarySrtsDetails>::template ReduceTileWithCarry<REDUCE_INTO_CARRY>(
			srts_details.secondary_details, carry, reduction_op);
	}


	/**
	 * Reduction in SRTS grid.  Result is computed in all threads.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ T ReduceTile(
		SrtsDetails srts_details,
		ReductionOp reduction_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, reduction_op);

			// Place partial in next grid
			srts_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively reduce in next grid
		return CooperativeGridReduction<SecondarySrtsDetails>::ReduceTile(
			srts_details.secondary_details, reduction_op);
	}
};



} // namespace reduction
} // namespace util
} // namespace b40c

