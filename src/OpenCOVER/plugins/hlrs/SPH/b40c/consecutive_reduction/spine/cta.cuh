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
 * CTA-processing functionality for consecutive reduction spine scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>

namespace b40c {
namespace consecutive_reduction {
namespace spine {


/**
 * Consecutive reduction spine scan CTA
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::ValueType 		ValueType;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::SpineSizeT 		SpineSizeT;

	typedef typename KernelPolicy::SrtsSoaDetails 	SrtsSoaDetails;
	typedef typename KernelPolicy::TileTuple 		TileTuple;
	typedef typename KernelPolicy::SoaScanOperator	SoaScanOperator;

	typedef util::Tuple<
		ValueType (*)[KernelPolicy::LOAD_VEC_SIZE],
		SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]> 	TileSoa;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Operational details for SRTS grid
	SrtsSoaDetails 		srts_soa_details;

	// The tuple value we will accumulate (in raking threads only)
	TileTuple 			carry;

	// Device input/outputs
	ValueType 			*d_in_partials;
	ValueType 			*d_out_partials;

	// Output device pointer
	SizeT 				*d_in_flags;
	SizeT 				*d_out_flags;

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
		ValueType 			*d_in_partials,
		ValueType 			*d_out_partials,
		SizeT 				*d_in_flags,
		SizeT				*d_out_flags,
		SoaScanOperator		soa_scan_op) :

			srts_soa_details(
				typename SrtsSoaDetails::GridStorageSoa(
					smem_storage.partials_raking_elements,
					smem_storage.flags_raking_elements),
				typename SrtsSoaDetails::WarpscanSoa(
					smem_storage.partials_warpscan,
					smem_storage.flags_warpscan),
				soa_scan_op()),
			d_in_partials(d_in_partials),
			d_out_partials(d_out_partials),
			d_in_flags(d_in_flags),
			d_out_flags(d_out_flags),
			soa_scan_op(soa_scan_op)
	{}





	/**
	 * Process a single tile
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessTile(
		SpineSizeT cta_offset,
		SpineSizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		// Tiles of consecutive reduction elements and flags
		ValueType			partials[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];
		SizeT				flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile of partials
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
				partials,
				d_in_partials,
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
				d_in_flags,
				cta_offset,
				guarded_elements);

		// SOA-scan tile of tuple pairs
		util::scan::soa::CooperativeSoaTileScan<KernelPolicy::LOAD_VEC_SIZE>::template
			ScanTileWithCarry<!FIRST_TILE>(
				srts_soa_details,
				TileSoa(partials, flags),
				carry,							// Seed with carry, maintain carry in raking threads
				soa_scan_op);

		// Store tile of partials
		util::io::StoreTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::Store(
				partials,
				d_out_partials,
				cta_offset,
				guarded_elements);

		// Store tile of flags
		util::io::StoreTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::Store(
				flags,
				d_out_flags,
				cta_offset,
				guarded_elements);
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SpineSizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SpineSizeT cta_offset = work_limits.offset;

		if (cta_offset < work_limits.guarded_offset) {

			// Process at least one full tile of tile_elements (first tile)
			ProcessTile<true>(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;

			while (cta_offset < work_limits.guarded_offset) {
				// Process more full tiles (not first tile)
				ProcessTile<false>(cta_offset);
				cta_offset += KernelPolicy::TILE_ELEMENTS;
			}

			// Clean up last partial tile with guarded-io (not first tile)
			if (work_limits.guarded_elements) {
				ProcessTile<false>(
					cta_offset,
					work_limits.guarded_elements);
			}

		} else {

			// Clean up last partial tile with guarded-io (first tile)
			ProcessTile<true>(
				cta_offset,
				work_limits.guarded_elements);
		}
	}
};



} // namespace spine
} // namespace consecutive_reduction
} // namespace b40c

