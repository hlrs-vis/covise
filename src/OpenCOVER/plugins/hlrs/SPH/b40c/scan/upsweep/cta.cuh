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
 * CTA-processing functionality for scan upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>

#include <b40c/util/reduction/cooperative_reduction.cuh>

namespace b40c {
namespace scan {
namespace upsweep {


/**
 * Scan scan upsweep reduction CTA
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::T 					T;
	typedef typename KernelPolicy::SizeT 				SizeT;
	typedef typename KernelPolicy::ReductionOp 			ReductionOp;
	typedef typename KernelPolicy::IdentityOp 			IdentityOp;

	typedef typename KernelPolicy::SrtsDetails 			SrtsDetails;
	typedef typename KernelPolicy::SmemStorage			SmemStorage;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Running partial accumulated by the CTA over its tile-processing
	// lifetime (managed in each raking thread)
	T carry;

	// Input and output device pointers
	T *d_in;
	T *d_spine;

	// Scan operator
	ReductionOp scan_op;

	// Operational details for SRTS scan grid
	SrtsDetails srts_details;



	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	template <typename SmemStorage>
	__device__ __forceinline__ Cta(
		SmemStorage 		&smem_storage,
		T 					*d_in,
		T 					*d_spine,
		ReductionOp 		scan_op,
		IdentityOp 			identity_op) :

			srts_details(
				smem_storage.raking_elements,
				smem_storage.warpscan,
				identity_op()),
			d_in(d_in),
			d_spine(d_spine),
			scan_op(scan_op),
			carry(scan_op())
	{}


	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		// Tile of scan elements
		T	partials[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile of partials
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
				partials,
				d_in,
				cta_offset,
				guarded_elements);

		// SOA-reduce tile of tuple pairs
		util::reduction::CooperativeTileReduction<
			KernelPolicy::LOAD_VEC_SIZE>::template ReduceTileWithCarry<true>(		// Maintain carry in thread SrtsSoaDetails::CUMULATIVE_THREAD
				srts_details,
				partials,
				carry,																// Seed with carry
				scan_op);

		// Barrier to protect srts_details before next tile
		__syncthreads();
	}


	/**
	 * Stores final reduction to output
	 */
	__device__ __forceinline__ void OutputToSpine()
	{
		// Write output
		if (threadIdx.x == SrtsDetails::CUMULATIVE_THREAD) {

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

		// Process full tiles of tile_elements
		while (cta_offset < work_limits.guarded_offset) {
			ProcessTile(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-io
		if (work_limits.guarded_elements) {
			ProcessTile(
				cta_offset,
				work_limits.guarded_elements);
		}

		// Produce output in spine
		OutputToSpine();
	}
};


} // namespace upsweep
} // namespace scan
} // namespace b40c

