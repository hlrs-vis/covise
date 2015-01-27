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
 * CTA-processing functionality for consecutive removal downsweep
 * scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/scan/cooperative_scan.cuh>
#include <b40c/util/operators.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/load_tile_discontinuity.cuh>
#include <b40c/util/io/store_tile.cuh>
#include <b40c/util/io/two_phase_scatter_tile.cuh>

namespace b40c {
namespace consecutive_removal {
namespace downsweep {


/**
 * Consecutive removal downsweep scan CTA
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 			KeyType;
	typedef typename KernelPolicy::ValueType		ValueType;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::EqualityOp		EqualityOp;

	typedef typename KernelPolicy::LocalFlag		LocalFlag;			// Type for noting local discontinuities
	typedef typename KernelPolicy::RankType			RankType;			// Type for local SRTS prefix sum

	typedef typename KernelPolicy::SrtsDetails 		SrtsDetails;

	typedef typename KernelPolicy::SmemStorage 		SmemStorage;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Running accumulator for the number of discontinuities observed by
	// the CTA over its tile-processing lifetime (managed in each raking thread)
	SizeT			carry;

	// Device pointers
	KeyType 		*d_in_keys;
	KeyType			*d_out_keys;
	ValueType 		*d_in_values;
	ValueType 		*d_out_values;
	SizeT			*d_num_compacted;

	// Shared memory storage for the CTA
	SmemStorage		&smem_storage;

	// Operational details for SRTS scan grid
	SrtsDetails 	srts_details;

	// Equality operator
	EqualityOp		equality_op;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Two-phase scatter tile specialization
	 */
	template <
		bool FIRST_TILE,
		bool TWO_PHASE_SCATTER = KernelPolicy::TWO_PHASE_SCATTER>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		KeyType 	keys[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];
		ValueType	values[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		LocalFlag 	head_flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];		// Discontinuity head_flags
		RankType 	ranks[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];			// Local scatter offsets

		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Process tile
		 */
		template <typename Cta>
		__device__ __forceinline__ void Process(
			SizeT cta_offset,
			const SizeT &guarded_elements,
			Cta *cta)
		{
			// Load keys, initializing discontinuity head_flags
			util::io::LoadTileDiscontinuity<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER,
				KernelPolicy::CHECK_ALIGNMENT,
				KernelPolicy::CONSECUTIVE_SMEM_ASSIST,
				FIRST_TILE,
				false>::LoadValid(			// Do not set flag for first oob element
					cta->smem_storage.assist_scratch,
					keys,
					head_flags,
					cta->d_in_keys,
					cta_offset,
					guarded_elements,
					cta->equality_op);

			// Copy discontinuity head_flags into ranks
			util::io::InitializeTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(ranks, head_flags);

			// Scan tile of ranks
			util::Sum<RankType> scan_op;
			RankType unique_elements =
				util::scan::CooperativeTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTile(
					cta->srts_details,
					ranks,
					scan_op);

			// Barrier sync to protect smem exchange storage
			__syncthreads();

			// 2-phase scatter keys
			util::io::TwoPhaseScatterTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER,
				KernelPolicy::CHECK_ALIGNMENT>::Scatter(
					keys,
					head_flags,
					ranks,
					unique_elements,
					cta->smem_storage.key_exchange,
					cta->d_out_keys,
					cta->carry);

			if (!util::Equals<ValueType, util::NullType>::VALUE) {

				// Load values
				util::io::LoadTile<
					KernelPolicy::LOG_LOADS_PER_TILE,
					KernelPolicy::LOG_LOAD_VEC_SIZE,
					KernelPolicy::THREADS,
					KernelPolicy::READ_MODIFIER,
					KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
						values,
						cta->d_in_values,
						cta_offset,
						guarded_elements);

				// Barrier sync to protect smem exchange storage
				__syncthreads();

				// 2-phase scatter values
				util::io::TwoPhaseScatterTile<
					KernelPolicy::LOG_LOADS_PER_TILE,
					KernelPolicy::LOG_LOAD_VEC_SIZE,
					KernelPolicy::THREADS,
					KernelPolicy::WRITE_MODIFIER,
					KernelPolicy::CHECK_ALIGNMENT>::Scatter(
						values,
						head_flags,
						ranks,
						unique_elements,
						cta->smem_storage.value_exchange,
						cta->d_out_values,
						cta->carry);
			}

			// Barrier sync to protect smem exchange storage
			__syncthreads();

			// Update running discontinuity count for CTA
			cta->carry += unique_elements;
		}
	};


	/**
	 * Direct-scatter tile specialization
	 */
	template <bool FIRST_TILE>
	struct Tile<FIRST_TILE, false>
	{
		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		KeyType 	keys[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];
		ValueType	values[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		LocalFlag 	head_flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];		// Discontinuity head_flags
		RankType 	ranks[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];			// Local scatter offsets

		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Process tile
		 */
		template <typename Cta>
		__device__ __forceinline__ void Process(
			SizeT cta_offset,
			const SizeT &guarded_elements,
			Cta *cta)
		{
			// Load keys tile, initializing discontinuity head_flags
			util::io::LoadTileDiscontinuity<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER,
				KernelPolicy::CHECK_ALIGNMENT,
				KernelPolicy::CONSECUTIVE_SMEM_ASSIST,
				FIRST_TILE,
				false>::LoadValid(			// Do not set flag for first oob element
					cta->smem_storage.assist_scratch,
					keys,
					head_flags,
					cta->d_in_keys,
					cta_offset,
					guarded_elements,
					cta->equality_op);

			// Copy discontinuity head_flags into ranks
			util::io::InitializeTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(ranks, head_flags);

			// Scan tile of ranks, seed with carry (maintain carry in raking threads)
			util::Sum<RankType> scan_op;
			util::scan::CooperativeTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTileWithCarry(
				cta->srts_details,
				ranks,
				cta->carry,
				scan_op);

			// Scatter valid keys directly to global output, predicated on head_flags
			util::io::ScatterTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					cta->d_out_keys,
					keys,
					head_flags,
					ranks);

			if (!util::Equals<ValueType, util::NullType>::VALUE) {

				// Load values
				util::io::LoadTile<
					KernelPolicy::LOG_LOADS_PER_TILE,
					KernelPolicy::LOG_LOAD_VEC_SIZE,
					KernelPolicy::THREADS,
					KernelPolicy::READ_MODIFIER,
					KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
						values,
						cta->d_in_values,
						cta_offset,
						guarded_elements);

				// Scatter valid values directly to global output, predicated on head_flags
				util::io::ScatterTile<
					KernelPolicy::LOG_LOADS_PER_TILE,
					KernelPolicy::LOG_LOAD_VEC_SIZE,
					KernelPolicy::THREADS,
					KernelPolicy::WRITE_MODIFIER>::Scatter(
						cta->d_out_values,
						values,
						head_flags,
						ranks);
			}
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage &smem_storage,
		KeyType 		*d_in_keys,
		KeyType 		*d_out_keys,
		ValueType 		*d_in_values,
		ValueType 		*d_out_values,
		SizeT			*d_num_compacted,
		EqualityOp		equality_op,
		SizeT			spine_partial = 0) :

			srts_details(
				smem_storage.raking_elements,
				smem_storage.warpscan,
				0),
			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_out_keys(d_out_keys),
			d_in_values(d_in_values),
			d_out_values(d_out_values),
			d_num_compacted(d_num_compacted),
			equality_op(equality_op),
			carry(spine_partial) 			// Seed carry with spine partial
	{}


	/**
	 * Process a single tile
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<FIRST_TILE> tile;

		tile.Process(cta_offset, guarded_elements, this);
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

		// Output number of compacted items
		if (work_limits.last_block && (threadIdx.x == 0)) {
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry, d_num_compacted);
		}
	}

};


} // namespace downsweep
} // namespace consecutive_removal
} // namespace b40c

