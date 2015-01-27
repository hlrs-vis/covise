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
 * CTA-processing functionality for consecutive reduction downsweep
 * scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/load_tile_discontinuity.cuh>
#include <b40c/util/io/store_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>

namespace b40c {
namespace consecutive_reduction {
namespace downsweep {


/**
 * Consecutive reduction downsweep scan CTA
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 				KeyType;
	typedef typename KernelPolicy::ValueType			ValueType;
	typedef typename KernelPolicy::SizeT 				SizeT;
	typedef typename KernelPolicy::EqualityOp			EqualityOp;

	typedef typename KernelPolicy::SpineSoaTuple 		SpineSoaTuple;

	typedef typename KernelPolicy::LocalFlag			LocalFlag;			// Type for noting local discontinuities
	typedef typename KernelPolicy::RankType				RankType;			// Type for local SRTS prefix sum

	typedef typename KernelPolicy::SrtsSoaDetails 		SrtsSoaDetails;
	typedef typename KernelPolicy::TileTuple 			TileTuple;
	typedef typename KernelPolicy::SoaScanOperator		SoaScanOperator;

	typedef util::Tuple<
		ValueType (*)[KernelPolicy::LOAD_VEC_SIZE],
		RankType (*)[KernelPolicy::LOAD_VEC_SIZE]> 		TileSoa;

	typedef typename KernelPolicy::SmemStorage 			SmemStorage;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Operational details for SRTS grid
	SrtsSoaDetails 		srts_soa_details;

	// The spine value-flag tuple value we will accumulate (in raking threads only)
	SpineSoaTuple 		carry;

	// Device pointers
	KeyType 			*d_in_keys;
	KeyType				*d_out_keys;
	ValueType 			*d_in_values;
	ValueType 			*d_out_values;
	SizeT				*d_num_compacted;

	// Operators
	SoaScanOperator 	soa_scan_op;
	EqualityOp			equality_op;

	SmemStorage			&smem_storage;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile (direct-scatter specialization)
	 */
	template <
		bool FIRST_TILE,
		bool TWO_PHASE_SCATTER = KernelPolicy::TWO_PHASE_SCATTER>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Typedefs and Constants
		//---------------------------------------------------------------------

		enum {
			LOADS_PER_TILE = KernelPolicy::LOADS_PER_TILE,
			LOAD_VEC_SIZE = KernelPolicy::LOAD_VEC_SIZE,
		};


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		KeyType			keys[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];
		ValueType		values[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		LocalFlag		head_flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];		// Tile of discontinuity flags
		RankType 		ranks[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];			// Tile of global scatter offsets


		//---------------------------------------------------------------------
		// Helper structures
		//---------------------------------------------------------------------

		/**
		 * Decrement transform
		 */
		template <typename T>
		struct DecrementOp
		{
			__device__ __forceinline__ T operator()(T data)
			{
				return data - 1;
			}
		};

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
				true>::LoadValid(			// Set flag for first oob element
					cta->smem_storage.assist_scratch,
					keys,
					head_flags,
					cta->d_in_keys,
					cta_offset,
					guarded_elements,
					cta->equality_op);

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

			// Copy discontinuity head_flags into ranks
			util::io::InitializeTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(ranks, head_flags);

			// SOA-scan tile of tuple pairs
			if (FIRST_TILE && (blockIdx.x == 0)) {

				// A single-CTA or first-CTA does not incorporate a seed partial from carry
				// on the first tile (because either the spine does not exist or the first
				// spine element is invalid)
				util::scan::soa::CooperativeSoaTileScan<KernelPolicy::LOAD_VEC_SIZE>::template
					ScanTileWithCarry<false>(				// Assign carry
						cta->srts_soa_details,
						TileSoa(values, ranks),
						cta->carry,							// maintain carry in raking threads
						cta->soa_scan_op);
			} else {

				// Seed the soa scan with carry
				util::scan::soa::CooperativeSoaTileScan<KernelPolicy::LOAD_VEC_SIZE>::template
					ScanTileWithCarry<true>(				// Update carry
						cta->srts_soa_details,
						TileSoa(values, ranks),
						cta->carry,							// Seed with carry, maintain carry in raking threads
						cta->soa_scan_op);
			}

			// Scatter valid keys directly to global output, predicated on head_flags
			util::io::ScatterTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					cta->d_out_keys,
					keys,
					head_flags,
					ranks,
					guarded_elements);						// We explicitly want to restrict by guarded_elements

			// Decrement scatter ranks for values
			util::io::InitializeTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE>::Transform(
					ranks, ranks, DecrementOp<RankType>());

			// First CTA unsets the first head flag of first tile
			if (FIRST_TILE && (blockIdx.x == 0) && (threadIdx.x == 0)) {
				head_flags[0][0] = 0;
			}

			// Scatter valid reduced values directly to global output, predicated on head_flags
			util::io::ScatterTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					cta->d_out_values,
					values,
					head_flags,
					ranks);									// We explicitly do not want to restrict by guarded_elements
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		KeyType 		*d_out_keys,
		ValueType 		*d_in_values,
		ValueType 		*d_out_values,
		SizeT			*d_num_compacted,
		SoaScanOperator		soa_scan_op,
		EqualityOp		equality_op) :

			smem_storage(smem_storage),
			srts_soa_details(
				typename SrtsSoaDetails::GridStorageSoa(
					smem_storage.partials_raking_elements,
					smem_storage.ranks_raking_elements),
				typename SrtsSoaDetails::WarpscanSoa(
					smem_storage.partials_warpscan,
					smem_storage.ranks_warpscan),
				soa_scan_op()),
			d_in_keys(d_in_keys),
			d_out_keys(d_out_keys),
			d_in_values(d_in_values),
			d_out_values(d_out_values),
			d_num_compacted(d_num_compacted),
			soa_scan_op(soa_scan_op),
			equality_op(equality_op)
	{}


	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 		&smem_storage,
		KeyType 			*d_in_keys,
		KeyType 			*d_out_keys,
		ValueType 			*d_in_values,
		ValueType 			*d_out_values,
		SizeT				*d_num_compacted,
		SoaScanOperator		soa_scan_op,
		EqualityOp			equality_op,
		SpineSoaTuple		spine_partial) :

			smem_storage(smem_storage),
			srts_soa_details(
				typename SrtsSoaDetails::GridStorageSoa(
					smem_storage.partials_raking_elements,
					smem_storage.ranks_raking_elements),
				typename SrtsSoaDetails::WarpscanSoa(
					smem_storage.partials_warpscan,
					smem_storage.ranks_warpscan),
				soa_scan_op()),
			d_in_keys(d_in_keys),
			d_out_keys(d_out_keys),
			d_in_values(d_in_values),
			d_out_values(d_out_values),
			d_num_compacted(d_num_compacted),
			soa_scan_op(soa_scan_op),
			equality_op(equality_op),
			carry(spine_partial)				// Seed carry with spine partial
	{}


	/**
	 * Process a single tile
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<FIRST_TILE> tile;

		tile.Process(cta_offset, guarded_elements, this);
	}


	/**
	 * Process work range of tiles
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessLastTile(
		util::CtaWorkLimits<SizeT> &work_limits,
		SizeT cta_offset)
	{
		// Clean up last partial tile with guarded-io if necessary
		if (work_limits.guarded_elements) {

			ProcessTile<FIRST_TILE>(
				cta_offset,
				work_limits.guarded_elements);

			// Output the number of compacted items
			if (threadIdx.x == SrtsSoaDetails::CUMULATIVE_THREAD) {
				util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
					carry.t1 - 1, d_num_compacted);
			}

		} else if ((work_limits.last_block) && (threadIdx.x == SrtsSoaDetails::CUMULATIVE_THREAD)) {

			// Partial-tile processing outputs the final reduced value.  If there is
			// no partial work for the last CTA, it must instead write the final reduced value
			// residing in its carry.t1 flag
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry.t0,
				d_out_values + carry.t1 - 1);

			// Output the number of compacted items
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry.t1, d_num_compacted);
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

			// Process at least one full tile of tile_elements (first tile)
			ProcessTile<true>(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;

			while (cta_offset < work_limits.guarded_offset) {
				// Process more full tiles (not first tile)
				ProcessTile<false>(cta_offset);
				cta_offset += KernelPolicy::TILE_ELEMENTS;
			}

			// Process last (partial) tile (not first tile)
			ProcessLastTile<false>(work_limits, cta_offset);

		} else {

			// Process last (partial) tile (first tile)
			ProcessLastTile<true>(work_limits, cta_offset);
		}
	}
};


} // namespace downsweep
} // namespace consecutive_reduction
} // namespace b40c

