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
 * CTA-processing functionality for scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

#include <b40c/util/scan/cooperative_scan.cuh>

namespace b40c {
namespace scan {
namespace downsweep {


/**
 * Scan downsweep scan CTA
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::T 			T;
	typedef typename KernelPolicy::SizeT 		SizeT;
	typedef typename KernelPolicy::ReductionOp 	ReductionOp;
	typedef typename KernelPolicy::IdentityOp 	IdentityOp;

	typedef typename KernelPolicy::SrtsDetails 	SrtsDetails;
	typedef typename KernelPolicy::SmemStorage	SmemStorage;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Running partial accumulated by the CTA over its tile-processing
	// lifetime (managed in each raking thread)
	T carry;

	// Input and output device pointers
	T *d_in;
	T *d_out;

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
	__device__ __forceinline__ Cta(
		SmemStorage 		&smem_storage,
		T 					*d_in,
		T 					*d_out,
		ReductionOp 		scan_op,
		IdentityOp 			identity_op) :

			srts_details(
				smem_storage.RakingElements(),
				smem_storage.warpscan,
				identity_op()),
			d_in(d_in),
			d_out(d_out),
			scan_op(scan_op),
			carry(identity_op()) {}			// Seed carry with identity

	/**
	 * Constructor with spine partial for seeding with
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 		&smem_storage,
		T 					*d_in,
		T 					*d_out,
		ReductionOp 		scan_op,
		IdentityOp 			identity_op,
		T 					spine_partial) :

			srts_details(
				smem_storage.RakingElements(),
				smem_storage.warpscan,
				identity_op()),
			d_in(d_in),
			d_out(d_out),
			scan_op(scan_op),
			carry(spine_partial) {}			// Seed carry with spine partial


	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		// Tile of scan elements
		T partials[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile
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

		// Scan tile with carry update in raking threads
		util::scan::CooperativeTileScan<
			KernelPolicy::LOAD_VEC_SIZE,
			KernelPolicy::EXCLUSIVE>::ScanTileWithCarry(
				srts_details,
				partials,
				carry,
				scan_op);

		// Store tile
		util::io::StoreTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::Store(
				partials,
				d_out,
				cta_offset,
				guarded_elements);
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
	}

};


} // namespace downsweep
} // namespace scan
} // namespace b40c

