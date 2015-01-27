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
 * Tile-processing functionality for BFS compaction upsweep kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/scan/cooperative_scan.cuh>

#include <b40c/util/operators.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace microbench {
namespace local_cull {


texture<char, cudaTextureType1D, cudaReadModeElementType> bitmask_tex_ref;


/**
 * Cta
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 		VertexId;
	typedef typename KernelPolicy::ValidFlag		ValidFlag;
	typedef typename KernelPolicy::CollisionMask 	CollisionMask;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::ThreadId			ThreadId;
	typedef typename KernelPolicy::SrtsDetails 		SrtsDetails;
	typedef typename KernelPolicy::SmemStorage		SmemStorage;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Current BFS iteration
	VertexId 				iteration;

	// Current BFS queue index
	VertexId 				queue_index;

	// Input and output device pointers
	VertexId 				*d_in;						// Incoming vertex ids
	SizeT 					*d_out_row_offsets;			// Compacted row offsets
	SizeT 					*d_out_row_lengths;			// Compacted row lengths
	CollisionMask 			*d_collision_cache;
	SizeT					*d_row_offsets;
	VertexId				*d_source_path;

	// Work progress
	util::CtaWorkProgress	&work_progress;

	// Operational details for SRTS scan grid
	SrtsDetails 			srts_details;

	SmemStorage				&smem_storage;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile
	 */
	template <
		int LOG_LOADS_PER_TILE,
		int LOG_LOAD_VEC_SIZE>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Typedefs and Constants
		//---------------------------------------------------------------------

		enum {
			LOADS_PER_TILE 		= 1 << LOG_LOADS_PER_TILE,
			LOAD_VEC_SIZE 		= 1 << LOG_LOAD_VEC_SIZE
		};


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		// Dequeued vertex ids
		VertexId 	vertex_ids[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_lengths[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_offsets[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Whether or not the corresponding vertex_ids is valid for exploring
		ValidFlag 	flags[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Tile of local scatter offsets
		SizeT 		ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];

		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate over vertex id
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * InitFlags
			 */
			static __device__ __forceinline__ void InitFlags(Tile *tile)
			{
				// Initially valid if vertex-id is valid
				tile->flags[LOAD][VEC] = (tile->vertex_ids[LOAD][VEC] == -1) ? 0 : 1;

				// Next
				Iterate<LOAD, VEC + 1>::InitFlags(tile);
			}


			/**
			 * BitmaskCull
			 */
			static __device__ __forceinline__ void BitmaskCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					// Location of mask byte to read
					SizeT mask_byte_offset = (tile->vertex_ids[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) >> 3;

					// Read byte from from collision cache bitmask
					CollisionMask mask_byte = tex1Dfetch(
						bitmask_tex_ref,
						mask_byte_offset);

					// Bit in mask byte corresponding to current vertex id
					CollisionMask mask_bit = 1 << (tile->vertex_ids[LOAD][VEC] & 7);

					if (mask_bit & mask_byte) {

						// seen it in mask
						tile->flags[LOAD][VEC] = 0;

					} else {

						util::io::ModifiedLoad<util::io::ld::cg>::Ld(
							mask_byte, cta->d_collision_cache + mask_byte_offset);

						if (mask_bit & mask_byte) {

							// seen it in mask
							tile->flags[LOAD][VEC] = 0;

						} else {

							// Update with best effort
							mask_byte |= mask_bit;
							util::io::ModifiedStore<util::io::st::cg>::St(
								mask_byte,
								cta->d_collision_cache + mask_byte_offset);
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::BitmaskCull(cta, tile);
			}


			/**
			 * HistoryCull
			 */
			static __device__ __forceinline__ void HistoryCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					int hash = ((typename KernelPolicy::UnsignedBits) tile->vertex_ids[LOAD][VEC]) % SmemStorage::HISTORY_HASH_ELEMENTS;
					VertexId retrieved = cta->smem_storage.history[hash];

					if (retrieved == tile->vertex_ids[LOAD][VEC]) {
						// Seen it
						tile->flags[LOAD][VEC] = 0;

					} else {
						// Update it
						cta->smem_storage.history[hash] = tile->vertex_ids[LOAD][VEC];
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::HistoryCull(cta, tile);
			}


			/**
			 * WarpCull
			 */
			static __device__ __forceinline__ void WarpCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					int warp_id 		= threadIdx.x >> 5;
					int hash 			= tile->vertex_ids[LOAD][VEC] & (SmemStorage::WARP_HASH_ELEMENTS - 1);

					cta->smem_storage.state.vid_hashtable[warp_id][hash] = tile->vertex_ids[LOAD][VEC];
					VertexId retrieved = cta->smem_storage.state.vid_hashtable[warp_id][hash];

					if (retrieved == tile->vertex_ids[LOAD][VEC]) {

						cta->smem_storage.state.vid_hashtable[warp_id][hash] = threadIdx.x;
						VertexId tid = cta->smem_storage.state.vid_hashtable[warp_id][hash];
						if (tid != threadIdx.x) {
							tile->flags[LOAD][VEC] = 0;
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::WarpCull(cta, tile);
			}


			/**
			 * LabelCull
			 */
			static __device__ __forceinline__ void LabelCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					VertexId row_id = tile->vertex_ids[LOAD][VEC];

					// Load source path of node
					VertexId source_path;
					util::io::ModifiedLoad<util::io::ld::cg>::Ld(
						source_path,
						cta->d_source_path + row_id);


					if (source_path != -1) {

						// Seen it
						tile->flags[LOAD][VEC] = 0;

					} else {

						// Update source path with current iteration
						util::io::ModifiedStore<util::io::st::cg>::St(
							cta->iteration,
							cta->d_source_path + row_id);
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::LabelCull(cta, tile);
			}


			/**
			 * AtomicCull
			 */
			static __device__ __forceinline__ void AtomicCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					VertexId distance = atomicCAS(
						cta->d_source_path + tile->vertex_ids[LOAD][VEC],
						-1,
						cta->iteration);

					if (distance == -1) {

						// Unvisited: get row offset/length
						tile->row_offsets[LOAD][VEC] = cta->d_row_offsets[tile->vertex_ids[LOAD][VEC]];
						tile->row_lengths[LOAD][VEC] = cta->d_row_offsets[tile->vertex_ids[LOAD][VEC] + 1] -
							tile->row_offsets[LOAD][VEC];

					} else {
						// Visited
						tile->flags[LOAD][VEC] = 0;
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::AtomicCull(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			// InitFlags
			static __device__ __forceinline__ void InitFlags(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::InitFlags(tile);
			}

			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::BitmaskCull(cta, tile);
			}

			// HistoryCull
			static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::HistoryCull(cta, tile);
			}

			// WarpCull
			static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::WarpCull(cta, tile);
			}

			// LabelCull
			static __device__ __forceinline__ void LabelCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::LabelCull(cta, tile);
			}

			// AtomicCull
			static __device__ __forceinline__ void AtomicCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::AtomicCull(cta, tile);
			}
		};



		/**
		 * Terminate iteration
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// InitFlags
			static __device__ __forceinline__ void InitFlags(Tile *tile) {}

			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile) {}

			// HistoryCull
			static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile) {}

			// WarpCull
			static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile) {}

			// LabelCull
			static __device__ __forceinline__ void LabelCull(Cta *cta, Tile *tile) {}

			// AtomicCull
			static __device__ __forceinline__ void AtomicCull(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Initializer
		 */
		__device__ __forceinline__ void InitFlags()
		{
			Iterate<0, 0>::InitFlags(this);
		}

		/**
		 * Culls vertices based upon whether or not we've set a bit for them
		 * in the d_collision_cache bitmask
		 */
		__device__ __forceinline__ void BitmaskCull(Cta *cta)
		{
			Iterate<0, 0>::BitmaskCull(cta, this);
		}

		/**
		 * Culls vertices based upon local duplicate collisions
		 */
		__device__ __forceinline__ void HistoryCull(Cta *cta)
		{
			Iterate<0, 0>::HistoryCull(cta, this);
		}

		/**
		 * Does label vertex culling
		 */
		__device__ __forceinline__ void WarpCull(Cta *cta)
		{
			Iterate<0, 0>::WarpCull(cta, this);
		}

		/**
		 * Does label vertex culling
		 */
		__device__ __forceinline__ void LabelCull(Cta *cta)
		{
			Iterate<0, 0>::LabelCull(cta, this);
		}

		/**
		 * Does perfect vertex culling
		 */
		__device__ __forceinline__ void AtomicCull(Cta *cta)
		{
			Iterate<0, 0>::AtomicCull(cta, this);
		}
	};




	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		VertexId 				iteration,
		VertexId				queue_index,
		SmemStorage 			&smem_storage,
		VertexId 				*d_in,
		SizeT 					*d_out_row_offsets,
		SizeT 					*d_out_row_lengths,
		CollisionMask 			*d_collision_cache,
		SizeT					*d_row_offsets,
		VertexId				*d_source_path,
		util::CtaWorkProgress	&work_progress) :

			iteration(iteration),
			queue_index(queue_index),
			smem_storage(smem_storage),
			srts_details(
				smem_storage.state.raking_elements,
				smem_storage.state.warpscan,
				0),
			d_in(d_in),
			d_out_row_offsets(d_out_row_offsets),
			d_out_row_lengths(d_out_row_lengths),
			d_collision_cache(d_collision_cache),
			d_row_offsets(d_row_offsets),
			d_source_path(d_source_path),
			work_progress(work_progress)

	{
		// Initialize history duplicate-filter
		for (int offset = threadIdx.x; offset < SmemStorage::HISTORY_HASH_ELEMENTS; offset += KernelPolicy::THREADS) {
			smem_storage.history[offset] = -1;
		}
	}


	/**
	 * Process a single, full tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			false>::LoadValid(
				tile.vertex_ids,
				d_in,
				cta_offset,
				guarded_elements,
				(VertexId) -1);

		// Init valid flags
		tile.InitFlags();

		if (KernelPolicy::BENCHMARK) {

			// Cull using global collision bitmask
			tile.BitmaskCull(this);

			tile.LabelCull(this);

			// Cull using local collision hashing
			tile.WarpCull(this);

			tile.HistoryCull(this);

		} else {

			// Cull using atomic CAS
			tile.AtomicCull(this);
		}

		// Copy flags into ranks
		util::io::InitializeTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(tile.ranks, tile.flags);

		// Protect repurposable storage that backs both raking lanes and local cull scratch
		__syncthreads();

		// Scan tile of ranks, using an atomic add to reserve
		// space in the compacted queue, seeding ranks
		util::Sum<SizeT> scan_op;
		util::scan::CooperativeTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTileWithEnqueue(
			srts_details,
			tile.ranks,
			work_progress.GetQueueCounter<SizeT>(queue_index + 1),
			scan_op);

		// Protect repurposable storage that backs both raking lanes and local cull scratch
		__syncthreads();

		if (!KernelPolicy::BENCHMARK) {

			// Scatter valid row offsets
			util::io::ScatterTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					d_out_row_offsets,
					tile.row_offsets,
					tile.flags,
					tile.ranks);

			// Scatter valid row lengths
			util::io::ScatterTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					d_out_row_lengths,
					tile.row_lengths,
					tile.flags,
					tile.ranks);
		}
	}
};


} // namespace local_cull
} // namespace microbench
} // namespace bfs
} // namespace graph
} // namespace b40c

