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
 ******************************************************************************/

/******************************************************************************
 * CTA-processing functionality for consecutive removal upsweep
 * reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>
#include <b40c/util/operators.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/load_tile_discontinuity.cuh>

namespace b40c {
namespace consecutive_removal {
namespace upsweep {


/**
 * Consecutive removal upsweep reduction CTA
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 				KeyType;
	typedef typename KernelPolicy::ValueType			ValueType;
	typedef typename KernelPolicy::SizeT 				SizeT;
	typedef typename KernelPolicy::EqualityOp			EqualityOp;

	typedef int 										LocalFlag;		// Type for noting local discontinuities (just needs to count up to TILE_ELEMENTS_PER_THREAD)
	typedef typename KernelPolicy::SmemStorage 			SmemStorage;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Accumulator for the number of discontinuities observed (in each thread)
	SizeT			carry;

	// Device pointers
	KeyType 		*d_in_keys;
	SizeT			*d_spine;

	// Shared memory storage for the CTA
	SmemStorage		&smem_storage;

	// Equality operator
	EqualityOp		equality_op;



	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType			*d_in_keys,
		SizeT 			*d_spine,
		EqualityOp		equality_op) :

			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_spine(d_spine),
			equality_op(equality_op),
			carry(0)
	{}


	/**
	 * Process a single, full tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		KeyType		keys[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];
		LocalFlag 	head_flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];		// Tile of discontinuity head_flags

		// Load data tile, initializing discontinuity head_flags
		util::io::LoadTileDiscontinuity<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT,
			KernelPolicy::CONSECUTIVE_SMEM_ASSIST,
			FIRST_TILE,
			false>::LoadValid(			// Do not set flag for first oob element
				smem_storage.assist_scratch,
				keys,
				head_flags,
				d_in_keys,
				cta_offset,
				guarded_elements,
				equality_op);

		// Prevent accumulation from being hoisted (otherwise we don't get the desired outstanding loads)
		if (KernelPolicy::LOADS_PER_TILE > 1) __syncthreads();

		// Reduce head_flags, accumulate in carry
		carry += util::reduction::SerialReduce<KernelPolicy::TILE_ELEMENTS_PER_THREAD>::Invoke(
			(LocalFlag*) head_flags);
	}


	/**
	 * Collective reduction across all threads, stores final reduction to output
	 *
	 * Used to collectively reduce each thread's aggregate after striding through
	 * the input.
	 */
	__device__ __forceinline__ void OutputToSpine()
	{
		// Cooperatively reduce the carries in each thread (thread-0 gets the result)
		util::Sum<SizeT> reduction_op;
		carry = util::reduction::TreeReduce<KernelPolicy::LOG_THREADS, false>::Invoke(				// No need to return aggregate reduction in all threads
			carry,
			smem_storage.reduction_tree,
			reduction_op);

		// Write output
		if (threadIdx.x == 0) {
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry, d_spine + blockIdx.x);
		}
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT cta_offset = work_limits.offset;

		if (cta_offset < work_limits.guarded_offset) {

			// Process at least one full tile of tile_elements
			ProcessTile<true>(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;

			// Process more full tiles (not first tile)
			while (cta_offset < work_limits.guarded_offset) {
				ProcessTile<false>(cta_offset);
				cta_offset += KernelPolicy::TILE_ELEMENTS;
			}

			// Clean up last partial tile with guarded-io (not first tile)
			if (work_limits.guarded_elements) {
				ProcessTile<false>(
					cta_offset,
					work_limits.out_of_bounds);
			}

		} else {

			// Clean up last partial tile with guarded-io (first tile)
			ProcessTile<true>(
				cta_offset,
				work_limits.out_of_bounds);
		}

		// Collectively reduce accumulated carry from each thread into output
		// destination
		OutputToSpine();
	}
};



} // namespace upsweep
} // namespace consecutive_removal
} // namespace b40c

