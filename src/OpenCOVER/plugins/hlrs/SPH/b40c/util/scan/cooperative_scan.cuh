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
 * Cooperative tile reduction and scanning within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/cooperative_reduction.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/scan/warp_scan.cuh>

namespace b40c {
namespace util {
namespace scan {

/**
 * Cooperative reduction in SRTS smem grid hierarchies
 */
template <
	typename SrtsDetails,
	typename SecondarySrtsDetails = typename SrtsDetails::SecondarySrtsDetails>
struct CooperativeGridScan;



/**
 * Cooperative tile scan
 */
template <
	int VEC_SIZE,							// Length of vector-loads (e.g, vec-1, vec-2, vec-4)
	bool EXCLUSIVE = true>					// Whether or not this is an exclusive scan
struct CooperativeTileScan
{
	//---------------------------------------------------------------------
	// Iteration structures for extracting partials from SRTS lanes and
	// using them to seed scans of tile vectors
	//---------------------------------------------------------------------

	// Next lane/load
	template <int LANE, int TOTAL_LANES>
	struct ScanLane
	{
		template <typename SrtsDetails, typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			SrtsDetails srts_details,
			typename SrtsDetails::T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
			ReductionOp scan_op)
		{
			// Retrieve partial reduction from SRTS grid
			typename SrtsDetails::T exclusive_partial = srts_details.lane_partial[LANE][0];

			// Scan the partials in this lane/load
			SerialScan<VEC_SIZE, EXCLUSIVE>::Invoke(
				data[LANE], exclusive_partial, scan_op);

			// Next load
			ScanLane<LANE + 1, TOTAL_LANES>::Invoke(
				srts_details, data, scan_op);
		}
	};

	// Terminate
	template <int TOTAL_LANES>
	struct ScanLane<TOTAL_LANES, TOTAL_LANES>
	{
		template <typename SrtsDetails, typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			SrtsDetails srts_details,
			typename SrtsDetails::T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
			ReductionOp scan_op) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Scan a single tile.  Total aggregate is computed and returned in all threads.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <typename SrtsDetails, typename ReductionOp>
	static __device__ __forceinline__ typename SrtsDetails::T ScanTile(
		SrtsDetails srts_details,
		typename SrtsDetails::T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
		ReductionOp scan_op)
	{
		// Reduce partials in each vector-load, placing resulting partial in SRTS smem grid lanes (one lane per load)
		reduction::CooperativeTileReduction<VEC_SIZE>::template ReduceLane<0, SrtsDetails::SCAN_LANES>::Invoke(
			srts_details, data, scan_op);

		__syncthreads();

		CooperativeGridScan<SrtsDetails>::ScanTile(
			srts_details, scan_op);

		__syncthreads();

		// Scan each vector-load, seeded with the resulting partial from its SRTS grid lane,
		ScanLane<0, SrtsDetails::SCAN_LANES>::Invoke(
			srts_details, data, scan_op);

		// Return last thread's inclusive partial
		return srts_details.CumulativePartial();
	}

	/**
	 * Scan a single tile where carry is updated with the total aggregate only
	 * in raking threads.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <typename SrtsDetails, typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithCarry(
		SrtsDetails srts_details,
		typename SrtsDetails::T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
		typename SrtsDetails::T &carry,
		ReductionOp scan_op)
	{
		// Reduce partials in each vector-load, placing resulting partials in SRTS smem grid lanes (one lane per load)
		reduction::CooperativeTileReduction<VEC_SIZE>::template ReduceLane<0, SrtsDetails::SCAN_LANES>::Invoke(
			srts_details, data, scan_op);

		__syncthreads();

		CooperativeGridScan<SrtsDetails>::ScanTileWithCarry(
			srts_details, carry, scan_op);

		__syncthreads();

		// Scan each vector-load, seeded with the resulting partial from its SRTS grid lane,
		ScanLane<0, SrtsDetails::SCAN_LANES>::Invoke(
			srts_details, data, scan_op);
	}


	/**
	 * Scan a single tile with atomic enqueue.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <typename SrtsDetails, typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithEnqueue(
		SrtsDetails srts_details,
		typename SrtsDetails::T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
		typename SrtsDetails::T* d_enqueue_counter,
		ReductionOp scan_op)
	{
		// Reduce partials in each vector-load, placing resulting partial in SRTS smem grid lanes (one lane per load)
		reduction::CooperativeTileReduction<VEC_SIZE>::template ReduceLane<0, SrtsDetails::SCAN_LANES>::Invoke(
			srts_details, data, scan_op);

		__syncthreads();

		CooperativeGridScan<SrtsDetails>::ScanTileWithEnqueue(
			srts_details, d_enqueue_counter, scan_op);

		__syncthreads();

		// Scan each vector-load, seeded with the resulting partial from its SRTS grid lane,
		ScanLane<0, SrtsDetails::SCAN_LANES>::Invoke(
			srts_details, data, scan_op);
	}


	/**
	 * Scan a single tile with atomic enqueue.  Total aggregate is computed and
	 * returned in all threads.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <typename SrtsDetails, typename ReductionOp>
	static __device__ __forceinline__ typename SrtsDetails::T ScanTileWithEnqueue(
		SrtsDetails srts_details,
		typename SrtsDetails::T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
		typename SrtsDetails::T *d_enqueue_counter,
		typename SrtsDetails::T &enqueue_offset,
		ReductionOp scan_op)
	{
		// Reduce partials in each vector-load, placing resulting partial in SRTS smem grid lanes (one lane per load)
		reduction::CooperativeTileReduction<VEC_SIZE>::template ReduceLane<0, SrtsDetails::SCAN_LANES>::Invoke(
			srts_details, data, scan_op);

		__syncthreads();

		CooperativeGridScan<SrtsDetails>::ScanTileWithEnqueue(
			srts_details, d_enqueue_counter, enqueue_offset, scan_op);

		__syncthreads();

		// Scan each vector-load, seeded with the resulting partial from its SRTS grid lane,
		ScanLane<0, SrtsDetails::SCAN_LANES>::Invoke(
			srts_details, data, scan_op);

		// Return last thread's inclusive partial
		return srts_details.CumulativePartial();
	}
};




/******************************************************************************
 * CooperativeGridScan
 ******************************************************************************/

/**
 * Cooperative SRTS grid reduction (specialized for last-level of SRTS grid)
 */
template <typename SrtsDetails>
struct CooperativeGridScan<SrtsDetails, NullType>
{
	typedef typename SrtsDetails::T T;

	/**
	 * Scan in last-level SRTS grid.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTile(
		SrtsDetails srts_details,
		ReductionOp scan_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, scan_op);

			// Exclusive warp scan
			T exclusive_partial = WarpScan<SrtsDetails::LOG_RAKING_THREADS>::Invoke(
				inclusive_partial, srts_details.warpscan, scan_op);

			// Exclusive raking scan
			SerialScan<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, exclusive_partial, scan_op);

		}
	}


	/**
	 * Scan in last-level SRTS grid.  Carry-in/out is updated only in raking threads
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithCarry(
		SrtsDetails srts_details,
		T &carry,
		ReductionOp scan_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, scan_op);

			// Exclusive warp scan, get total
			T warpscan_total;
			T exclusive_partial = WarpScan<SrtsDetails::LOG_RAKING_THREADS>::Invoke(
				inclusive_partial, warpscan_total, srts_details.warpscan, scan_op);

			// Seed exclusive partial with carry-in
			exclusive_partial = scan_op(carry, exclusive_partial);

			// Exclusive raking scan
			SerialScan<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, exclusive_partial, scan_op);

			// Update carry
			carry = scan_op(carry, warpscan_total);			// Increment the CTA's running total by the full tile reduction
		}
	}


	/**
	 * Scan in last-level SRTS grid with atomic enqueue
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithEnqueue(
		SrtsDetails srts_details,
		T *d_enqueue_counter,
		ReductionOp scan_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, scan_op);

			// Exclusive warp scan, get total
			T warpscan_total;
			T exclusive_partial = WarpScan<SrtsDetails::LOG_RAKING_THREADS>::Invoke(
					inclusive_partial, warpscan_total, srts_details.warpscan, scan_op);

			// Atomic-increment the global counter with the total allocation
			T reservation_offset;
			if (threadIdx.x == 0) {
				reservation_offset = util::AtomicInt<T>::Add(
					d_enqueue_counter,
					warpscan_total);
				srts_details.warpscan[1][0] = reservation_offset;
			}

			// Seed exclusive partial with queue reservation offset
			reservation_offset = srts_details.warpscan[1][0];
			exclusive_partial = scan_op(reservation_offset, exclusive_partial);

			// Exclusive raking scan
			SerialScan<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, exclusive_partial, scan_op);
		}
	}

	/**
	 * Scan in last-level SRTS grid with atomic enqueue
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithEnqueue(
		SrtsDetails srts_details,
		T *d_enqueue_counter,
		T &enqueue_offset,
		ReductionOp scan_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, scan_op);

			// Exclusive warp scan, get total
			T warpscan_total;
			T exclusive_partial = WarpScan<SrtsDetails::LOG_RAKING_THREADS>::Invoke(
				inclusive_partial, warpscan_total, srts_details.warpscan, scan_op);

			// Atomic-increment the global counter with the total allocation
			if (threadIdx.x == 0) {
				enqueue_offset = util::AtomicInt<T>::Add(
					d_enqueue_counter,
					warpscan_total);
			}

			// Exclusive raking scan
			SerialScan<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, exclusive_partial, scan_op);
		}
	}
};


/**
 * Cooperative SRTS grid reduction for multi-level SRTS grids
 */
template <
	typename SrtsDetails,
	typename SecondarySrtsDetails>
struct CooperativeGridScan
{
	typedef typename SrtsDetails::T T;

	/**
	 * Scan in SRTS grid.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTile(
		SrtsDetails srts_details,
		ReductionOp scan_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = reduction::SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, scan_op);

			// Place partial in next grid
			srts_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively scan in next grid
		CooperativeGridScan<SecondarySrtsDetails>::ScanTile(
			srts_details.secondary_details, scan_op);

		__syncthreads();

		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Retrieve partial from next grid
			T exclusive_partial = srts_details.secondary_details.lane_partial[0][0];

			// Exclusive raking scan
			SerialScan<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, exclusive_partial, scan_op);
		}
	}

	/**
	 * Scan in SRTS grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithCarry(
		SrtsDetails srts_details,
		T &carry,
		ReductionOp scan_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = reduction::SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, scan_op);

			// Place partial in next grid
			srts_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively scan in next grid
		CooperativeGridScan<SecondarySrtsDetails>::ScanTileWithCarry(
			srts_details.secondary_details, carry, scan_op);

		__syncthreads();

		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Retrieve partial from next grid
			T exclusive_partial = srts_details.secondary_details.lane_partial[0][0];

			// Exclusive raking scan
			SerialScan<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, exclusive_partial, scan_op);
		}
	}

	/**
	 * Scan in SRTS grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithEnqueue(
		SrtsDetails srts_details,
		T* d_enqueue_counter,
		ReductionOp scan_op)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = reduction::SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, scan_op);

			// Place partial in next grid
			srts_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively scan in next grid
		CooperativeGridScan<SecondarySrtsDetails>::ScanTileWithEnqueue(
			srts_details.secondary_details, d_enqueue_counter, scan_op);

		__syncthreads();

		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Retrieve partial from next grid
			T exclusive_partial = srts_details.secondary_details.lane_partial[0][0];

			// Exclusive raking scan
			SerialScan<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, exclusive_partial, scan_op);
		}
	}
};



} // namespace scan
} // namespace util
} // namespace b40c

