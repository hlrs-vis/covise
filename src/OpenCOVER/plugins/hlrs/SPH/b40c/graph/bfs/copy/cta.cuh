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
 * Tile-processing functionality for BFS copy kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace copy {


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
	typedef typename KernelPolicy::SizeT 			SizeT;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Current BFS iteration
	VertexId 				iteration;
	int 					num_gpus;

	// Input and output device pointers
	VertexId 				*d_in;
	VertexId 				*d_out;
	VertexId 				*d_parent_in;
	VertexId				*d_source_path;

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
		VertexId 	vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];
		VertexId 	parent_id[LOADS_PER_TILE][LOAD_VEC_SIZE];


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
			 * MarkSource
			 */
			static __device__ __forceinline__ void MarkSource(
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
				Iterate<LOAD, VEC + 1>::MarkSource(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			// MarkSource
			static __device__ __forceinline__ void MarkSource(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::MarkSource(cta, tile);
			}
		};



		/**
		 * Terminate iteration
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// MarkSource
			static __device__ __forceinline__ void MarkSource(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * MarkSource
		 */
		__device__ __forceinline__ void MarkSource(Cta *cta)
		{
			Iterate<0, 0>::MarkSource(cta, this);
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
		int						num_gpus,
		VertexId 				*d_in,
		VertexId 				*d_out,
		VertexId 				*d_parent_in,
		VertexId 				*d_source_path) :
			iteration(iteration),
			num_gpus(num_gpus),
			d_in(d_in),
			d_out(d_out),
			d_parent_in(d_parent_in),
			d_source_path(d_source_path)
	{}


	/**
	 * Process a single tile
	 *
	 * Each thread copies only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		// Tile of elements
		Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

		// Load tile of vertices
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			false>::LoadValid(
				tile.vertex_id,
				d_in,
				cta_offset,
				guarded_elements,
				(VertexId) -1);

		if (KernelPolicy::MARK_PARENTS) {

			// Load tile of parents
			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER,
				false>::LoadValid(
					tile.parent_id,
					d_parent_in,
					cta_offset,
					guarded_elements);
		}

		if (KernelPolicy::LOADS_PER_TILE > 1) __syncthreads();

		// Mark sources
		tile.MarkSource(this);

		// Store tile of vertices
		util::io::StoreTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER,
			false>::Store(
				tile.vertex_id,
				d_out,
				cta_offset,
				guarded_elements);
	}
};

} // namespace copy
} // namespace bfs
} // namespace graph
} // namespace b40c

