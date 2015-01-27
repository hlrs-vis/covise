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
 * Tile-processing functionality for BFS compact-expand kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/scan/cooperative_scan.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>
#include <b40c/util/operators.cuh>

#include <b40c/graph/bfs/expand_atomic/cta.cuh>
#include <b40c/graph/bfs/compact_atomic/cta.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace compact_expand_atomic {


/**
 * Templated texture reference for collision bitmask
 */
template <typename CollisionMask>
struct BitmaskTex
{
	static texture<CollisionMask, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename CollisionMask>
texture<CollisionMask, cudaTextureType1D, cudaReadModeElementType> BitmaskTex<CollisionMask>::ref;


/**
 * Templated texture reference for row-offsets
 */
template <typename SizeT>
struct RowOffsetTex
{
	static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename SizeT>
texture<SizeT, cudaTextureType1D, cudaReadModeElementType> RowOffsetTex<SizeT>::ref;



/**
 * Derivation of KernelPolicy that encapsulates tile-processing routines
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 		VertexId;
	typedef typename KernelPolicy::CollisionMask 	CollisionMask;
	typedef typename KernelPolicy::SizeT 			SizeT;

	typedef typename KernelPolicy::SmemStorage		SmemStorage;

	typedef typename KernelPolicy::SoaScanOp		SoaScanOp;
	typedef typename KernelPolicy::SrtsSoaDetails 	SrtsSoaDetails;
	typedef typename KernelPolicy::TileTuple 		TileTuple;

	typedef util::Tuple<
		SizeT (*)[KernelPolicy::LOAD_VEC_SIZE],
		SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]> 	RankSoa;



	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Current BFS iteration
	VertexId 				iteration;
	VertexId 				queue_index;
	int 					num_gpus;

	// Input and output device pointers
	VertexId 				*d_in;
	VertexId 				*d_out;
	VertexId 				*d_parent_in;
	VertexId 				*d_parent_out;
	VertexId				*d_column_indices;
	SizeT					*d_row_offsets;
	VertexId				*d_source_path;
	CollisionMask			*d_collision_cache;

	// Work progress
	util::CtaWorkProgress	&work_progress;

	// Operational details for SRTS grid
	SrtsSoaDetails 			srts_soa_details;

	// Shared memory for the CTA
	SmemStorage				&smem_storage;

	bool 					bitmask_cull;


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

		typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		// Dequeued vertex ids
		VertexId 	vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];
		VertexId 	parent_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Edge list details
		SizeT		row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Temporary state for local culling
		int 		hash[LOADS_PER_TILE][LOAD_VEC_SIZE];			// Hash ids for vertex ids
		bool 		duplicate[LOADS_PER_TILE][LOAD_VEC_SIZE];		// Status as potential duplicate

		SizeT 		fine_count;
		SizeT		progress;

		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate next vector element
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * Inspect
			 */
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					// Translate vertex-id into local gpu row-id (currently stride of num_gpu)
					VertexId row_id = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) / cta->num_gpus;

					// Node is previously unvisited: compute row offset and length
					tile->row_offset[LOAD][VEC] = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id);
					tile->row_length[LOAD][VEC] = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id + 1) - tile->row_offset[LOAD][VEC];
				}

				tile->fine_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
					tile->row_length[LOAD][VEC] : 0;

				tile->coarse_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
					0 : tile->row_length[LOAD][VEC];

				Iterate<LOAD, VEC + 1>::Inspect(cta, tile);
			}


			/**
			 * BitmaskCull
			 */
			static __device__ __forceinline__ void BitmaskCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					// Location of mask byte to read
					SizeT mask_byte_offset = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) >> 3;

					// Bit in mask byte corresponding to current vertex id
					CollisionMask mask_bit = 1 << (tile->vertex_id[LOAD][VEC] & 7);

					// Read byte from from collision cache bitmask tex
					CollisionMask mask_byte = tex1Dfetch(
						BitmaskTex<CollisionMask>::ref,
						mask_byte_offset);

					if (mask_bit & mask_byte) {

						// Seen it
						tile->vertex_id[LOAD][VEC] = -1;

					} else {

						util::io::ModifiedLoad<util::io::ld::cg>::Ld(
							mask_byte, cta->d_collision_cache + mask_byte_offset);

						if (mask_bit & mask_byte) {

							// Seen it
							tile->vertex_id[LOAD][VEC] = -1;

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
			 * VertexCull
			 */
			static __device__ __forceinline__ void VertexCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					VertexId row_id = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) / cta->num_gpus;

					// Load source path of node
					VertexId source_path;
					util::io::ModifiedLoad<util::io::ld::cg>::Ld(
						source_path,
						cta->d_source_path + row_id);


					if (source_path != -1) {

						// Seen it
						tile->vertex_id[LOAD][VEC] = -1;

					} else {

						if (KernelPolicy::MARK_PARENTS) {

							// Update source path with parent vertex
							util::io::ModifiedStore<util::io::st::cg>::St(
								tile->parent_id[LOAD][VEC],
								cta->d_source_path + row_id);
						} else {

							// Update source path with current iteration
							util::io::ModifiedStore<util::io::st::cg>::St(
								cta->iteration,
								cta->d_source_path + row_id);
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::VertexCull(cta, tile);
			}


			/**
			 * CtaCull
			 */
			__device__ __forceinline__ void CtaCull(
				Cta *cta,
				Tile *tile)
			{
				// Hash the node-IDs into smem scratch

				int hash = tile->vertex_id[LOAD][VEC] % SmemStorage::HASH_ELEMENTS;
				bool duplicate = false;

				// Hash the node-IDs into smem scratch
				if (tile->vertex_id[LOAD][VEC] != -1) {
					cta->smem_storage.cta_hashtable[hash] = tile->vertex_id[LOAD][VEC];
				}

				__syncthreads();

				// Retrieve what vertices "won" at the hash locations. If a
				// different node beat us to this hash cell; we must assume
				// that we may not be a duplicate.  Otherwise assume that
				// we are a duplicate... for now.

				if (tile->vertex_id[LOAD][VEC] != -1) {
					VertexId hashed_node = cta->smem_storage.cta_hashtable[hash];
					duplicate = (hashed_node == tile->vertex_id[LOAD][VEC]);
				}

				__syncthreads();

				// For the possible-duplicates, hash in thread-IDs to select
				// one of the threads to be the unique one
				if (duplicate) {
					cta->smem_storage.cta_hashtable[hash] = threadIdx.x;
				}

				__syncthreads();

				// See if our thread won out amongst everyone with similar node-IDs
				if (duplicate) {
					// If not equal to our tid, we are not an authoritative thread
					// for this node-ID
					if (cta->smem_storage.cta_hashtable[hash] != threadIdx.x) {
						tile->vertex_id[LOAD][VEC] = -1;
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::CtaCull(cta, tile);
			}


			/**
			 * WarpCull
			 */
			static __device__ __forceinline__ void WarpCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					int warp_id 		= threadIdx.x >> 5;
					int hash 			= tile->vertex_id[LOAD][VEC] & (SmemStorage::WARP_HASH_ELEMENTS - 1);

					cta->smem_storage.warp_hashtable[warp_id][hash] = tile->vertex_id[LOAD][VEC];
					VertexId retrieved = cta->smem_storage.warp_hashtable[warp_id][hash];

					if (retrieved == tile->vertex_id[LOAD][VEC]) {

						cta->smem_storage.warp_hashtable[warp_id][hash] = threadIdx.x;
						VertexId tid = cta->smem_storage.warp_hashtable[warp_id][hash];
						if (tid != threadIdx.x) {
							tile->vertex_id[LOAD][VEC] = -1;
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::WarpCull(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			/**
			 * Inspect
			 */
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Inspect(cta, tile);
			}

			/**
			 * BitmaskCull
			 */
			static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::BitmaskCull(cta, tile);
			}

			// VertexCull
			static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::VertexCull(cta, tile);
			}

			/**
			 * WarpCull
			 */
			static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::WarpCull(cta, tile);
			}

			/**
			 * CtaCull
			 */
			static __device__ __forceinline__ void CtaCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::CtaCull(cta, tile);
			}
		};

		/**
		 * Terminate
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// Inspect
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {}

			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile) {}

			// VertexCull
			static __device__ __forceinline__ void VertexCull(Cta *cta, Tile *tile) {}

			// WarpCull
			static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile) {}

			// CtaCull
			static __device__ __forceinline__ void CtaCull(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Initializer
		 */
		__device__ __forceinline__ void Init()
		{
			expand_atomic::Cta<KernelPolicy>::template Tile<
				LOG_LOADS_PER_TILE,
				LOG_LOAD_VEC_SIZE>::template Iterate<0, 0>::Init(this);
		}

		/**
		 * Inspect dequeued vertices, updating source path if necessary and
		 * obtaining edge-list details
		 */
		__device__ __forceinline__ void Inspect(Cta *cta)
		{
			Iterate<0, 0>::Inspect(cta, this);
		}

		/**
		 * Expands neighbor lists for valid vertices at CTA-expansion granularity
		 */
		__device__ __forceinline__ void ExpandByCta(Cta *cta)
		{
			expand_atomic::Cta<KernelPolicy>::template Tile<
				LOG_LOADS_PER_TILE,
				LOG_LOAD_VEC_SIZE>::template Iterate<0, 0>::ExpandByCta(cta, this);
		}

		/**
		 * Expands neighbor lists for valid vertices a warp-expansion granularity
		 */
		__device__ __forceinline__ void ExpandByWarp(Cta *cta)
		{
			expand_atomic::Cta<KernelPolicy>::template Tile<
				LOG_LOADS_PER_TILE,
				LOG_LOAD_VEC_SIZE>::template Iterate<0, 0>::ExpandByWarp(cta, this);
		}

		/**
		 * Expands neighbor lists by local scan rank
		 */
		__device__ __forceinline__ void ExpandByScan(Cta *cta)
		{
			expand_atomic::Cta<KernelPolicy>::template Tile<
				LOG_LOADS_PER_TILE,
				LOG_LOAD_VEC_SIZE>::template Iterate<0, 0>::ExpandByScan(cta, this);
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
		 * Culls vertices
		 */
		__device__ __forceinline__ void VertexCull(Cta *cta)
		{
			Iterate<0, 0>::VertexCull(cta, this);
		}

		/**
		 * Warp cull
		 */
		__device__ __forceinline__ void WarpCull(Cta *cta)
		{
			Iterate<0, 0>::WarpCull(cta, this);

			__syncthreads();
		}

		/**
		 * CTA cull
		 */
		__device__ __forceinline__ void CtaCull(Cta *cta)
		{
			Iterate<0, 0>::WarpCull(cta, this);

			__syncthreads();
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
		VertexId 				*d_out,
		VertexId 				*d_parent_in,
		VertexId 				*d_parent_out,
		VertexId 				*d_column_indices,
		SizeT 					*d_row_offsets,
		VertexId 				*d_source_path,
		CollisionMask 			*d_collision_cache,
		util::CtaWorkProgress	&work_progress) :

			srts_soa_details(
				typename SrtsSoaDetails::GridStorageSoa(
					smem_storage.coarse_raking_elements,
					smem_storage.fine_raking_elements),
				typename SrtsSoaDetails::WarpscanSoa(
					smem_storage.state.coarse_warpscan,
					smem_storage.state.fine_warpscan),
				TileTuple(0, 0)),
			smem_storage(smem_storage),
			iteration(iteration),
			queue_index(queue_index),
			num_gpus(1),
			d_in(d_in),
			d_out(d_out),
			d_parent_in(d_parent_in),
			d_parent_out(d_parent_out),
			d_column_indices(d_column_indices),
			d_row_offsets(d_row_offsets),
			d_source_path(d_source_path),
			d_collision_cache(d_collision_cache),
			work_progress(work_progress),
			bitmask_cull((KernelPolicy::BITMASK_CULL_THRESHOLD >= 0) && (smem_storage.state.work_decomposition.num_elements > KernelPolicy::TILE_ELEMENTS * KernelPolicy::BITMASK_CULL_THRESHOLD * ((SizeT) gridDim.x)))
	{}



	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE> tile;
		tile.Init();

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::QUEUE_READ_MODIFIER,
			false>::LoadValid(
				tile.vertex_id,
				d_in,
				cta_offset,
				guarded_elements,
				(VertexId) -1);

		// Load tile of parents
		if (KernelPolicy::MARK_PARENTS) {

			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::QUEUE_READ_MODIFIER,
				false>::LoadValid(
					tile.parent_id,
					d_parent_in,
					cta_offset,
					guarded_elements);
		}

		// Cull using global collision bitmask
		if (bitmask_cull) {
			tile.BitmaskCull(this);
		}

		// Cull using vertex visitation status
		tile.VertexCull(this);

		// Cull valid flags using local collision hashing
		tile.WarpCull(this);
//		tile.CtaCull(this);

		// Inspect dequeued vertices, updating source path and obtaining
		// edge-list details
		tile.Inspect(this);

		// Scan tile with carry update in raking threads
		SoaScanOp scan_op;
		TileTuple totals;
		util::scan::soa::CooperativeSoaTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTile(
			totals,
			srts_soa_details,
			RankSoa(tile.coarse_row_rank, tile.fine_row_rank),
			scan_op);

		SizeT coarse_count = totals.t0;
		tile.fine_count = totals.t1;

		if (threadIdx.x == 0) {
			smem_storage.state.coarse_enqueue_offset = work_progress.Enqueue(
				coarse_count + tile.fine_count,
				queue_index + 1);
			smem_storage.state.fine_enqueue_offset = smem_storage.state.coarse_enqueue_offset + coarse_count;
		}

		// Enqueue valid edge lists into outgoing queue (includes barrier)
		tile.ExpandByCta(this);

		// Enqueue valid edge lists into outgoing queue
		tile.ExpandByWarp(this);

		//
		// Enqueue the adjacency lists of unvisited node-IDs by repeatedly
		// gathering edges into the scratch space, and then
		// having the entire CTA copy the scratch pool into the outgoing
		// frontier queue.
		//

		tile.progress = 0;
		while (tile.progress < tile.fine_count) {

			// Fill the scratch space with gather-offsets for neighbor-lists.
			tile.ExpandByScan(this);

			__syncthreads();

			// Copy scratch space into queue
			int scratch_remainder = B40C_MIN(SmemStorage::GATHER_ELEMENTS, tile.fine_count - tile.progress);

			for (int scratch_offset = threadIdx.x;
				scratch_offset < scratch_remainder;
				scratch_offset += KernelPolicy::THREADS)
			{
				// Gather a neighbor
				VertexId neighbor_id;
				util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
					neighbor_id,
					d_column_indices + smem_storage.gather_offsets[scratch_offset]);

				// Scatter it into queue
				util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
					neighbor_id,
					d_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);

				if (KernelPolicy::MARK_PARENTS) {
					// Scatter parent it into queue
					VertexId parent_id = smem_storage.gather_parents[scratch_offset];
					util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
						parent_id,
						d_parent_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
				}
			}

			tile.progress += SmemStorage::GATHER_ELEMENTS;

			__syncthreads();
		}
	}
};



} // namespace compact_expand_atomic
} // namespace bfs
} // namespace graph
} // namespace b40c

