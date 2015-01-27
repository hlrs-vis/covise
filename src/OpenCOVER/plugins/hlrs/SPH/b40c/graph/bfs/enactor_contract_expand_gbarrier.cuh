/******************************************************************************
 * Copyright 2010 Duane Merrill
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
 ******************************************************************************/

/******************************************************************************
 * Contract-expand, single-launch breadth-first-search enactor.
 ******************************************************************************/

#pragma once


#include <b40c/util/global_barrier.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/enactor_base.cuh>
#include <b40c/graph/bfs/problem_type.cuh>

#include <b40c/graph/bfs/compact_expand_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/compact_expand_atomic/kernel.cuh>


namespace b40c {
namespace graph {
namespace bfs {



/**
 * Contract-expand, single-launch breadth-first-search enactor.
 *
 * Performs all search iterations with single kernel launch (using
 * software global barriers).  For each BFS iteration, the kernel
 * culls visited vertices and expands neighbor lists in a
 * single tile-processing phase.
 */
class EnactorContractExpandGBarrier : public EnactorBase
{

protected:

	/**
	 * Mechanism for implementing software global barriers from within
	 * a single grid invocation
	 */
	util::GlobalBarrierLifetime 		global_barrier;

	/**
	 * CTA duty kernel stats
	 */
	util::KernelRuntimeStatsLifetime 	kernel_stats;

	unsigned long long 					total_runtimes;			// Total time "worked" by each cta
	unsigned long long 					total_lifetimes;		// Total time elapsed by each cta

	unsigned long long 					total_queued;

	/**
	 * Current iteration (mapped into GPU space so that it can
	 * be modified by multi-iteration kernel launches)
	 */
	volatile long long 					*iteration;
	long long 							*d_iteration;

public: 	
	
	/**
	 * Constructor
	 */
	EnactorContractExpandGBarrier(bool DEBUG = false) :
		EnactorBase(DEBUG),
		iteration(NULL),
		d_iteration(NULL)
	{}


	/**
	 * Destructor
	 */
	virtual ~EnactorContractExpandGBarrier()
	{
		if (iteration) util::B40CPerror(cudaFreeHost((void *) iteration), "EnactorContractExpandGBarrier cudaFreeHost iteration failed", __FILE__, __LINE__);
	}


	/**
	 * Search setup / lazy initialization
	 */
	cudaError_t Setup(int grid_size)
    {
    	cudaError_t retval = cudaSuccess;

		do {

			// Make sure iteration is initialized
			if (!iteration) {

				int flags = cudaHostAllocMapped;

				// Allocate pinned memory
				if (retval = util::B40CPerror(cudaHostAlloc((void **)&iteration, sizeof(long long) * 1, flags),
					"EnactorContractExpandGBarrier cudaHostAlloc iteration failed", __FILE__, __LINE__)) break;

				// Map into GPU space
				if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **)&d_iteration, (void *) iteration, 0),
					"EnactorContractExpandGBarrier cudaHostGetDevicePointer iteration failed", __FILE__, __LINE__)) break;
			}

			// Make sure barriers are initialized
			if (retval = global_barrier.Setup(grid_size)) break;

			// Make sure our runtime stats are initialized
			if (retval = kernel_stats.Setup(grid_size)) break;

			// Reset statistics
			iteration[0] 		= 0;
			total_runtimes 		= 0;
			total_lifetimes 	= 0;
			total_queued 		= 0;


		} while (0);

		return retval;
	}


    /**
     * Obtain statistics about the last BFS search enacted 
     */
	template <typename VertexId>
    void GetStatistics(
    	long long &total_queued,
    	VertexId &search_depth,
    	double &avg_duty)
    {
    	total_queued = this->total_queued;
    	search_depth = iteration[0] - 1;
    	avg_duty = (total_lifetimes > 0) ?
    		double(total_runtimes) / total_lifetimes :
    		0.0;
    }
    

	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <
    	typename KernelPolicy,
    	bool INSTRUMENT,
    	typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		int 							max_grid_size = 0)
	{
		typedef typename CsrProblem::SizeT 			SizeT;
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::CollisionMask 	CollisionMask;

		cudaError_t retval = cudaSuccess;

		do {

			// Determine grid size
			int occupancy = KernelPolicy::CTA_OCCUPANCY;
			int grid_size = MaxGridSize(occupancy, max_grid_size);

			if (DEBUG) printf("DEBUG: BFS occupancy %d, grid size %d\n", occupancy, grid_size);
			fflush(stdout);

			// Setup / lazy initialization
			if (retval = Setup(grid_size)) break;

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			// Bind bitmask texture
			int bytes = (graph_slice->nodes + 8 - 1) / 8;
			cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					compact_expand_atomic::BitmaskTex<CollisionMask>::ref,
					graph_slice->d_collision_cache,
					bitmask_desc,
					bytes),
				"EnactorContractExpandGBarrier cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

			// Bind row-offsets texture
			cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					compact_expand_atomic::RowOffsetTex<SizeT>::ref,
					graph_slice->d_row_offsets,
					row_offsets_desc,
					(graph_slice->nodes + 1) * sizeof(SizeT)),
				"EnactorContractExpandGBarrier cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

			// Initiate single-grid kernel
			compact_expand_atomic::KernelGlobalBarrier<KernelPolicy>
					<<<grid_size, KernelPolicy::THREADS>>>(
				0,												// iteration
				0,												// queue_index
				0,												// steal_index
				src,

				graph_slice->frontier_queues.d_keys[0],
				graph_slice->frontier_queues.d_keys[1],
				graph_slice->frontier_queues.d_values[0],
				graph_slice->frontier_queues.d_values[1],

				graph_slice->d_column_indices,
				graph_slice->d_row_offsets,
				graph_slice->d_source_path,
				graph_slice->d_collision_cache,
				this->work_progress,
				this->global_barrier,

				this->kernel_stats,
				(VertexId *) d_iteration);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "EnactorContractExpandGBarrier Kernel failed ", __FILE__, __LINE__))) break;

			if (INSTRUMENT) {
				// Get stats
				if (retval = kernel_stats.Accumulate(
					grid_size,
					total_runtimes,
					total_lifetimes,
					total_queued)) break;
			}

		} while (0);

		return retval;
	}

    /**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <bool INSTRUMENT, typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		int 							max_grid_size = 0)
	{
		if (this->cuda_props.device_sm_version >= 200) {

			// Single-grid tuning configuration
			typedef compact_expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
				8,						// CTA_OCCUPANCY
				7,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				0,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::cg,		// QUEUE_READ_MODIFIER,
				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
				util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
				util::io::st::cg,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				32,						// WARP_GATHER_THRESHOLD
				128 * 4, 				// CTA_GATHER_THRESHOLD,
				3,						// BITMASK_CULL_THRESHOLD
				6> KernelPolicy;

			return EnactSearch<KernelPolicy, INSTRUMENT, CsrProblem>(
				csr_problem, src, max_grid_size);

		} else if (this->cuda_props.device_sm_version >= 130) {
/*
			// Single-grid tuning configuration
			typedef compact_expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				130,
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
				1,						// CTA_OCCUPANCY
				8,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				1, 						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				32,						// WARP_GATHER_THRESHOLD
				128 * 4, 				// CTA_GATHER_THRESHOLD,
				3,						// BITMASK_CULL_THRESHOLD
				6> KernelPolicy;

			return EnactSearch<KernelPolicy, INSTRUMENT>(
				csr_problem, src, max_grid_size);
*/
		}

		printf("Not yet tuned for this architecture\n");
		return cudaErrorInvalidConfiguration;
	}
    
};



} // namespace bfs
} // namespace graph
} // namespace b40c
