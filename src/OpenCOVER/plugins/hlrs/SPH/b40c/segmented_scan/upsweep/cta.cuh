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
 * CTA-processing functionality for segmented scan upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/reduction/soa/cooperative_soa_reduction.cuh>

namespace b40c {
namespace segmented_scan {
namespace upsweep {


/**
 * Segmented scan upsweep reduction CTA
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::T 					T;
	typedef typename KernelPolicy::Flag 				Flag;
	typedef typename KernelPolicy::SizeT 				SizeT;

	typedef typename KernelPolicy::SrtsSoaDetails 		SrtsSoaDetails;
	typedef typename KernelPolicy::TileTuple 			TileTuple;
	typedef typename KernelPolicy::SoaScanOperator		SoaScanOperator;

	typedef util::Tuple<
		T (*)[KernelPolicy::LOAD_VEC_SIZE],
		Flag (*)[KernelPolicy::LOAD_VEC_SIZE]> TileSoa;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Operational details for SRTS grid
	SrtsSoaDetails 		srts_soa_details;

	// The tuple value we will accumulate (in SrtsDetails::CUMULATIVE_THREAD thread only)
	TileTuple 			carry;

	// Input device pointers
	T 					*d_partials_in;
	Flag 				*d_flags_in;

	// Spine output device pointers
	T 					*d_spine_partials;
	Flag 				*d_spine_flags;

	// Scan operator
	SoaScanOperator 	soa_scan_op;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	template <typename SmemStorage>
	__device__ __forceinline__ Cta(
		SmemStorage 		&smem_storage,
		T 					*d_partials_in,
		Flag 				*d_flags_in,
		T 					*d_spine_partials,
		Flag 				*d_spine_flags,
		SoaScanOperator 	soa_scan_op) :

			srts_soa_details(
				typename SrtsSoaDetails::GridStorageSoa(
					smem_storage.partials_raking_elements,
					smem_storage.flags_raking_elements),
				typename SrtsSoaDetails::WarpscanSoa(
					smem_storage.partials_warpscan,
					smem_storage.flags_warpscan),
				soa_scan_op()),
			d_partials_in(d_partials_in),
			d_flags_in(d_flags_in),
			d_spine_partials(d_spine_partials),
			d_spine_flags(d_spine_flags),
			soa_scan_op(soa_scan_op),
			carry(soa_scan_op())
	{}


	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		// Tile of segmented scan elements and flags
		T		partials[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];
		Flag	flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile of partials
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
				partials,
				d_partials_in,
				cta_offset,
				guarded_elements);

		// Load tile of flags
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
				flags,
				d_flags_in,
				cta_offset,
				guarded_elements);

		// SOA-reduce tile of tuple pairs
		util::reduction::soa::CooperativeSoaTileReduction<
			KernelPolicy::LOAD_VEC_SIZE>::template ReduceTileWithCarry<true>(		// Maintain carry in thread SrtsSoaDetails::CUMULATIVE_THREAD
				srts_soa_details,
				TileSoa(partials, flags),
				carry,																// Seed with carry
				soa_scan_op);

		// Barrier to protect srts_soa_details before next tile
		__syncthreads();
	}


	/**
	 * Stores final reduction to output
	 */
	__device__ __forceinline__ void OutputToSpine()
	{
		// Write output
		if (threadIdx.x == SrtsSoaDetails::CUMULATIVE_THREAD) {

			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry.t0, d_spine_partials + blockIdx.x);

			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry.t1, d_spine_flags + blockIdx.x);
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
} // namespace segmented_scan
} // namespace b40c

