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
 * Tile-processing functionality for BFS expansion kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/scan/cooperative_scan.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/operators.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace expand_compact_atomic {


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

	// Row-length cutoff below which we expand neighbors by writing gather
	// offsets into scratch space (instead of gang-pressing warps or the entire CTA)
	static const int SCAN_EXPAND_CUTOFF = B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH);

	typedef typename KernelPolicy::SmemStorage			SmemStorage;
	typedef typename KernelPolicy::VertexId 			VertexId;
	typedef typename KernelPolicy::SizeT 				SizeT;
	typedef typename KernelPolicy::CollisionMask 		CollisionMask;

	typedef typename KernelPolicy::SrtsExpandDetails 	SrtsExpandDetails;
	typedef typename KernelPolicy::SrtsCompactDetails 	SrtsCompactDetails;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Current BFS iteration
	VertexId 				iteration;
	VertexId 				queue_index;

	// Input and output device pointers
	VertexId 				*d_in;
	VertexId 				*d_out;
	VertexId 				*d_parent_in;
	VertexId 				*d_parent_out;
	VertexId				*d_column_indices;
	SizeT					*d_row_offsets;
	VertexId				*d_source_path;
	CollisionMask 			*d_collision_cache;

	// Work progress
	util::CtaWorkProgress	&work_progress;

	// Operational details for SRTS scan grid
	SrtsExpandDetails 		srts_expand_details;
	SrtsCompactDetails 		srts_compact_details;

	// Shared memory
	SmemStorage 			&smem_storage;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	/**
	 * BitmaskCull
	 */
	__device__ __forceinline__ void BitmaskCull(VertexId &neighbor_id)
	{
		if (neighbor_id != -1) {

			// Location of mask byte to read
			SizeT mask_byte_offset = (neighbor_id & KernelPolicy::VERTEX_ID_MASK) >> 3;

			// Bit in mask byte corresponding to current vertex id
			CollisionMask mask_bit = 1 << (neighbor_id & 7);

			// Read byte from from collision cache bitmask tex
			CollisionMask mask_byte = tex1Dfetch(
				BitmaskTex<CollisionMask>::ref,
				mask_byte_offset);

			if (mask_bit & mask_byte) {

				// Seen it
				neighbor_id = -1;

			} else {

				util::io::ModifiedLoad<util::io::ld::cg>::Ld(
					mask_byte, d_collision_cache + mask_byte_offset);

				if (mask_bit & mask_byte) {

					// Seen it
					neighbor_id = -1;

				} else {

					// Update with best effort
					mask_byte |= mask_bit;
					util::io::ModifiedStore<util::io::st::cg>::St(
						mask_byte,
						d_collision_cache + mask_byte_offset);
				}
			}
		}
	}


	/**
	 * VertexCull
	 */
	__device__ __forceinline__ void VertexCull(VertexId &neighbor_id)
	{
		if (neighbor_id != -1) {

			VertexId row_id = neighbor_id & KernelPolicy::VERTEX_ID_MASK;

			// Load source path of node
			VertexId source_path;
			util::io::ModifiedLoad<util::io::ld::cg>::Ld(
				source_path,
				d_source_path + row_id);

			if (source_path != -1) {

				// Seen it
				neighbor_id = -1;

			} else {

				if (KernelPolicy::MARK_PARENTS) {

					// MOOCH Update source path with parent vertex

				} else {

					// Update source path with current iteration
					util::io::ModifiedStore<util::io::st::cg>::St(
						iteration + 1,
						d_source_path + row_id);
				}
			}
		}
	}


	/**
	 * CtaCull
	 */
	__device__ __forceinline__ void CtaCull(VertexId &vertex)
	{
		// Hash the node-IDs into smem scratch

		int hash = vertex % SmemStorage::HASH_ELEMENTS;
		bool duplicate = false;

		// Hash the node-IDs into smem scratch
		if (vertex != -1) {
			smem_storage.cta_hashtable[hash] = vertex;
		}

		__syncthreads();

		// Retrieve what vertices "won" at the hash locations. If a
		// different node beat us to this hash cell; we must assume
		// that we may not be a duplicate.  Otherwise assume that
		// we are a duplicate... for now.

		if (vertex != -1) {
			VertexId hashed_node = smem_storage.cta_hashtable[hash];
			duplicate = (hashed_node == vertex);
		}

		__syncthreads();

		// For the possible-duplicates, hash in thread-IDs to select
		// one of the threads to be the unique one
		if (duplicate) {
			smem_storage.cta_hashtable[hash] = threadIdx.x;
		}

		__syncthreads();

		// See if our thread won out amongst everyone with similar node-IDs
		if (duplicate) {
			// If not equal to our tid, we are not an authoritative thread
			// for this node-ID
			if (smem_storage.cta_hashtable[hash] != threadIdx.x) {
				vertex = -1;
			}
		}
	}


	/**
	 * WarpCull
	 */
	__device__ __forceinline__ void WarpCull(VertexId &vertex)
	{
		if (vertex != -1) {

			int warp_id 		= threadIdx.x >> 5;
			int hash 			= vertex & (SmemStorage::WARP_HASH_ELEMENTS - 1);

			smem_storage.warp_hashtable[warp_id][hash] = vertex;
			VertexId retrieved = smem_storage.warp_hashtable[warp_id][hash];

			if (retrieved == vertex) {

				smem_storage.warp_hashtable[warp_id][hash] = threadIdx.x;
				VertexId tid = smem_storage.warp_hashtable[warp_id][hash];
				if (tid != threadIdx.x) {
					vertex = -1;
				}
			}
		}
	}


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
		SizeT		local_ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];
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
			 * Init
			 */
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				tile->row_length[LOAD][VEC] = 0;
				tile->row_progress[LOAD][VEC] = 0;

				Iterate<LOAD, VEC + 1>::Init(tile);
			}

			/**
			 * Inspect
			 */
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					// Translate vertex-id into local gpu row-id (currently stride of num_gpu)
					VertexId row_id = tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK;

					// Node is previously unvisited: compute row offset and length
					tile->row_offset[LOAD][VEC] = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id);
					tile->row_length[LOAD][VEC] = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id + 1) - tile->row_offset[LOAD][VEC];
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::Inspect(cta, tile);
			}


			/**
			 * Expand by CTA
			 */
			static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
			{
				// CTA-based expansion/loading
				while (true) {

					if (threadIdx.x < B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH)) {
						cta->smem_storage.state.cta_comm = KernelPolicy::THREADS;
					}

					__syncthreads();

					if (tile->row_length[LOAD][VEC] >= KernelPolicy::CTA_GATHER_THRESHOLD) {
						cta->smem_storage.state.cta_comm = threadIdx.x;
					}

					__syncthreads();

					int owner = cta->smem_storage.state.cta_comm;
					if (owner == KernelPolicy::THREADS) {
						break;
					}

					if (owner == threadIdx.x) {
						// Got control of the CTA
						cta->smem_storage.state.warp_comm[0][0] = tile->row_offset[LOAD][VEC];										// start
						cta->smem_storage.state.warp_comm[0][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob
						if (KernelPolicy::MARK_PARENTS) {
							cta->smem_storage.state.warp_comm[0][3] = tile->vertex_id[LOAD][VEC];									// parent
						}

						// Unset row length
						tile->row_length[LOAD][VEC] = 0;
					}

					__syncthreads();

					SizeT coop_offset 	= cta->smem_storage.state.warp_comm[0][0];
					SizeT coop_oob 		= cta->smem_storage.state.warp_comm[0][2];

					VertexId parent_id;
					if (KernelPolicy::MARK_PARENTS) {
						parent_id = cta->smem_storage.state.warp_comm[0][3];
					}

					while (coop_offset < coop_oob) {

						// Gather
						VertexId neighbor_id = -1;
						SizeT ranks[1][1] = { {0} };
						if (coop_offset + threadIdx.x < coop_oob) {

							util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
								neighbor_id, cta->d_column_indices + coop_offset + threadIdx.x);

							// Check
							if (cta->smem_storage.state.work_decomposition.num_elements > KernelPolicy::TILE_ELEMENTS * KernelPolicy::BITMASK_CULL_THRESHOLD * gridDim.x) {
								cta->BitmaskCull(neighbor_id);
							}
							cta->VertexCull(neighbor_id);

							if (neighbor_id != -1) ranks[0][0] = 1;
						}

						// Scan tile of ranks, using an atomic add to reserve
						// space in the compacted queue, seeding ranks
						util::Sum<SizeT> scan_op;
						util::scan::CooperativeTileScan<1>::ScanTileWithEnqueue(
							cta->srts_compact_details,
							ranks,
							cta->work_progress.GetQueueCounter<SizeT>(cta->queue_index + 1),
							scan_op);

						if (neighbor_id != -1) {

							// Scatter neighbor
							util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
								neighbor_id,
								cta->d_out + ranks[0][0]);

							if (KernelPolicy::MARK_PARENTS) {
								// Scatter parent
								util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
									parent_id,
									cta->d_parent_out + ranks[0][0]);
							}
						}

						coop_offset += KernelPolicy::THREADS;
					}
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByCta(cta, tile);
			}


			/**
			 * Expand by scan
			 */
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
			{
				// Attempt to make further progress on this dequeued item's neighbor
				// list if its current offset into local scratch is in range
				SizeT scratch_offset = tile->local_ranks[LOAD][VEC] + tile->row_progress[LOAD][VEC] - tile->progress;

				while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
					(scratch_offset < SmemStorage::OFFSET_ELEMENTS))
				{
					// Put gather offset into scratch space
					cta->smem_storage.offset_scratch[scratch_offset] = tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];

					if (KernelPolicy::MARK_PARENTS) {
						// Put dequeued vertex as the parent into scratch space
						cta->smem_storage.parent_scratch[scratch_offset] = tile->vertex_id[LOAD][VEC];
					}

					tile->row_progress[LOAD][VEC]++;
					scratch_offset++;
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByScan(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			/**
			 * Init
			 */
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Init(tile);
			}

			/**
			 * Inspect
			 */
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Inspect(cta, tile);
			}

			/**
			 * Expand by CTA
			 */
			static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByCta(cta, tile);
			}

			/**
			 * Expand by scan
			 */
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByScan(cta, tile);
			}
		};

		/**
		 * Terminate
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// Init
			static __device__ __forceinline__ void Init(Tile *tile) {}

			// Inspect
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {}

			// ExpandByCta
			static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile) {}

			// ExpandByScan
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Constructor
		 */
		__device__ __forceinline__ Tile()
		{
			Iterate<0, 0>::Init(this);
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
			Iterate<0, 0>::ExpandByCta(cta, this);
		}

		/**
		 * Expands neighbor lists by local scan rank
		 */
		__device__ __forceinline__ void ExpandByScan(Cta *cta)
		{
			Iterate<0, 0>::ExpandByScan(cta, this);
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
		VertexId 				queue_index,
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

			iteration(iteration),
			queue_index(queue_index),
			srts_expand_details(
				smem_storage.expand_raking_elements,
				smem_storage.state.warpscan,
				0),
			srts_compact_details(
				smem_storage.state.compact_raking_elements,
				smem_storage.state.warpscan,
				0),
			smem_storage(smem_storage),
			d_in(d_in),
			d_parent_in(d_parent_in),
			d_out(d_out),
			d_parent_out(d_parent_out),
			d_column_indices(d_column_indices),
			d_row_offsets(d_row_offsets),
			d_source_path(d_source_path),
			d_collision_cache(d_collision_cache),
			work_progress(work_progress) {}


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

//		CtaCull(tile.vertex_id[0][0]);
		WarpCull(tile.vertex_id[0][0]);


		// Inspect dequeued vertices, obtaining edge-list details
		tile.Inspect(this);

		// Enqueue valid edge lists into outgoing queue
		tile.ExpandByCta(this);

		// Copy lengths into ranks
		util::io::InitializeTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(tile.local_ranks, tile.row_length);

		// Scan tile of local ranks
		util::Sum<SizeT> scan_op;
		tile.fine_count = util::scan::CooperativeTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTile(
			srts_expand_details,
			tile.local_ranks,
			scan_op);

		__syncthreads();


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
			int scratch_remainder = B40C_MIN(SmemStorage::OFFSET_ELEMENTS, tile.fine_count - tile.progress);

			for (int scratch_offset = 0;
				scratch_offset < scratch_remainder;
				scratch_offset += KernelPolicy::THREADS)
			{
				// Gather a neighbor
				VertexId neighbor_id = -1;
				SizeT ranks[1][1] = { {0} };
				if (scratch_offset + threadIdx.x < scratch_remainder) {

					util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
						neighbor_id,
						d_column_indices + smem_storage.offset_scratch[scratch_offset + threadIdx.x]);

					// Check
					if (smem_storage.state.work_decomposition.num_elements > KernelPolicy::TILE_ELEMENTS * KernelPolicy::BITMASK_CULL_THRESHOLD * gridDim.x) {
						BitmaskCull(neighbor_id);
					}
					VertexCull(neighbor_id);

					if (neighbor_id != -1) ranks[0][0] = 1;
				}

				// Scan tile of ranks, using an atomic add to reserve
				// space in the compacted queue, seeding ranks
				util::Sum<SizeT> scan_op;
				util::scan::CooperativeTileScan<1>::ScanTileWithEnqueue(
					srts_compact_details,
					ranks,
					work_progress.GetQueueCounter<SizeT>(queue_index + 1),
					scan_op);

				if (neighbor_id != -1) {

					// Scatter it into queue
					util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
						neighbor_id,
						d_out + ranks[0][0]);

					if (KernelPolicy::MARK_PARENTS) {
						// Scatter parent it into queue
						VertexId parent_id = smem_storage.parent_scratch[scratch_offset + threadIdx.x];

						util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
							parent_id,
							d_parent_out + ranks[0][0]);
					}
				}
			}

			tile.progress += SmemStorage::OFFSET_ELEMENTS;

			__syncthreads();
		}
	}
};



} // namespace expand_compact_atomic
} // namespace bfs
} // namespace graph
} // namespace b40c

