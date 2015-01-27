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
 * Two-phase out-of-core BFS implementation (BFS level grid launch)
 ******************************************************************************/

#pragma once

#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/problem_type.cuh>
#include <b40c/graph/bfs/enactor_base.cuh>
#include <b40c/graph/bfs/microbench/local_cull/kernel.cuh>
#include <b40c/graph/bfs/microbench/local_cull/kernel_policy.cuh>
#include <b40c/graph/bfs/microbench/neighbor_gather/kernel.cuh>
#include <b40c/graph/bfs/microbench/neighbor_gather/kernel_policy.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace microbench {



/**
 * Microbenchmark enactor
 */
class EnactorCull : public EnactorBase
{

protected:

	/**
	 * CTA duty kernel stats
	 */
	util::KernelRuntimeStatsLifetime expand_kernel_stats;
	util::KernelRuntimeStatsLifetime compact_kernel_stats;

	unsigned long long 		expand_total_runtimes;			// Total time "worked" by each cta
	unsigned long long 		expand_total_lifetimes;			// Total time elapsed by each cta

	unsigned long long 		compact_total_runtimes;
	unsigned long long 		compact_total_lifetimes;


	long long 		total_unculled;
	long long 		search_depth;

	void *d_source_path2;

public: 	
	
	/**
	 * Constructor
	 */
	EnactorCull(bool DEBUG = false) :
		EnactorBase(DEBUG),
		search_depth(0),
		total_unculled(0),

		expand_total_runtimes(0),
		expand_total_lifetimes(0),
		compact_total_runtimes(0),
		compact_total_lifetimes(0),
		d_source_path2(0)
	{}


	/**
	 * Search setup / lazy initialization
	 */
	cudaError_t Setup(int expand_grid_size, int compact_grid_size)
    {
    	cudaError_t retval = cudaSuccess;

		do {

			// Make sure our runtime stats are good
			if (retval = expand_kernel_stats.Setup(expand_grid_size)) break;
			if (retval = compact_kernel_stats.Setup(compact_grid_size)) break;

			// Reset statistics
			total_unculled 		= 1;
			search_depth 		= 0;

		} while (0);

		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~EnactorCull()
	{
		if (d_source_path2) {
			cudaFree(d_source_path2);
		}
	}


    /**
     * Obtain statistics about the last BFS search enacted 
     */
	template <typename VertexId>
    void GetStatistics(
    	long long &total_unculled,
    	VertexId &search_depth,
    	double &expand_duty,
    	double &compact_duty)
    {
    	total_unculled = this->total_unculled;
    	search_depth = this->search_depth;

    	expand_duty = (expand_total_lifetimes > 0) ?
    		double(expand_total_runtimes) / expand_total_lifetimes :
    		0.0;

    	compact_duty = (compact_total_lifetimes > 0) ?
    		double(compact_total_runtimes) / compact_total_lifetimes :
    		0.0;
    }

    
	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <
    	typename ExpandPolicy,
    	typename CompactPolicy,
    	typename BenchCompactPolicy,
    	bool INSTRUMENT,
    	typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		typename CsrProblem::SizeT 		src_offset,
		typename CsrProblem::SizeT 		src_length,
		int 							max_grid_size = 0)
	{
		typedef typename CsrProblem::VertexId					VertexId;
		typedef typename CsrProblem::SizeT						SizeT;
		typedef typename CsrProblem::CollisionMask				CollisionMask;
		typedef typename CsrProblem::ValidFlag					ValidFlag;

		cudaError_t retval = cudaSuccess;

		do {
			// Determine grid size(s)
			int expand_min_occupancy 		= ExpandPolicy::CTA_OCCUPANCY;
			int expand_grid_size 			= MaxGridSize(expand_min_occupancy, max_grid_size);

			int compact_min_occupancy		= CompactPolicy::CTA_OCCUPANCY;
			int compact_grid_size 			= MaxGridSize(compact_min_occupancy, max_grid_size);

			if (DEBUG) printf("BFS expand min occupancy %d, level-grid size %d\n",
				expand_min_occupancy, expand_grid_size);
			if (DEBUG) printf("BFS compact min occupancy %d, level-grid size %d\n",
				compact_min_occupancy, compact_grid_size);

			printf("Iteration, Bitmask cull, Compaction queue, Expansion queue\n");
			printf("0, 1, 1, ");
			fflush(stdout);

			SizeT queue_length;
			VertexId iteration = 0;		// BFS iteration
			VertexId queue_index = 0;	// Work queue index
			VertexId steal_index = 0;	// Work stealing index
			int selector = 0;

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			// Setup / lazy initialization
			if (retval = Setup(expand_grid_size, compact_grid_size)) break;

			// Allocate extra copy of labels
			if (!d_source_path2) {
				if (retval = util::B40CPerror(
						cudaMalloc((void**) &d_source_path2,
						graph_slice->nodes * sizeof(VertexId)),
					"CsrProblem cudaMalloc d_source_path2 failed", __FILE__, __LINE__)) break;
			}

			// Allocate value queues if necessary
			if (!graph_slice->frontier_queues.d_values[0]) {
				if (retval = util::B40CPerror(
						cudaMalloc((void**) &graph_slice->frontier_queues.d_values[0],
						graph_slice->compact_queue_elements * sizeof(VertexId)),
					"CsrProblem cudaMalloc frontier_queues.d_values[0] failed", __FILE__, __LINE__)) break;
			}
			if (!graph_slice->frontier_queues.d_values[1]) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slice->frontier_queues.d_values[1],
						graph_slice->expand_queue_elements * sizeof(VertexId)),
					"CsrProblem cudaMalloc frontier_queues.d_values[1] failed", __FILE__, __LINE__)) break;
			}

			// Copy source offset
			if (retval = util::B40CPerror(cudaMemcpy(
					graph_slice->frontier_queues.d_keys[selector],				// d_in_row_offsets
					&src_offset,
					sizeof(SizeT) * 1,
					cudaMemcpyHostToDevice),
				"EnactorCull cudaMemcpy src_offset failed", __FILE__, __LINE__)) break;

			// Copy source length
			if (retval = util::B40CPerror(cudaMemcpy(
					graph_slice->frontier_queues.d_values[selector],			// d_in_row_lengths
					&src_length,
					sizeof(SizeT) * 1,
					cudaMemcpyHostToDevice),
				"EnactorCull cudaMemcpy src_offset failed", __FILE__, __LINE__)) break;

			// Copy source distance
			VertexId src_distance = 0;
			if (retval = util::B40CPerror(cudaMemcpy(
					graph_slice->d_source_path + src,
					&src_distance,
					sizeof(VertexId) * 1,
					cudaMemcpyHostToDevice),
				"EnactorCull cudaMemcpy src_offset failed", __FILE__, __LINE__)) break;

			// Initialize d_keep to reuse as alternate bitmask
			util::MemsetKernel<ValidFlag><<<128, 128, 0, graph_slice->stream>>>(
				graph_slice->d_keep,
				0,
				graph_slice->nodes);

			int bytes = (graph_slice->nodes + 8 - 1) / 8;
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<char>();

			while (true) {

				// Expansion
				neighbor_gather::Kernel<ExpandPolicy>
					<<<expand_grid_size, ExpandPolicy::THREADS>>>(
						iteration,
						queue_index,
						steal_index,
						graph_slice->frontier_queues.d_keys[selector],			// d_in_row_offsets
						graph_slice->frontier_queues.d_keys[selector ^ 1],		// d_out
						graph_slice->frontier_queues.d_values[selector],		// d_in_row_lengths
						graph_slice->d_column_indices,
						(CollisionMask *) graph_slice->d_keep,
						graph_slice->d_source_path,
						this->work_progress);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "neighbor_gather::Kernel failed ", __FILE__, __LINE__))) break;

				queue_index++;
				steal_index++;
				iteration++;

				// Get expansion queue length
				if (work_progress.GetQueueLength(queue_index, queue_length)) break;
				printf("%lld\n", (long long) queue_length);

				if (!queue_length) {
					// Done
					break;
				}

				// copy source distance
				if (retval = util::B40CPerror(cudaMemcpy(
						d_source_path2,
						graph_slice->d_source_path,
						sizeof(VertexId) * graph_slice->nodes,
						cudaMemcpyDeviceToDevice),
					"EnactorCull cudaMemcpy d_source_path2 failed", __FILE__, __LINE__)) break;

				if (util::B40CPerror(cudaBindTexture(
						0,
						local_cull::bitmask_tex_ref,
						graph_slice->d_collision_cache,
						channelDesc,
						bytes),
					"EnactorCull cudaBindTexture failed", __FILE__, __LINE__)) exit(1);

				// BenchCompaction
				local_cull::Kernel<BenchCompactPolicy>
					<<<compact_grid_size, BenchCompactPolicy::THREADS>>>(
						iteration,
						queue_index,
						steal_index,
						graph_slice->frontier_queues.d_keys[selector ^ 1],		// d_in
						graph_slice->frontier_queues.d_keys[selector],			// d_out_row_offsets
						graph_slice->frontier_queues.d_values[selector],		// d_out_row_lengths
						graph_slice->d_collision_cache,
						graph_slice->d_row_offsets,
						(VertexId *)d_source_path2,
						this->work_progress,
						this->compact_kernel_stats);

				if (INSTRUMENT) {
					// Get compact downsweep stats (i.e., duty %)
					if (retval = compact_kernel_stats.Accumulate(
						compact_grid_size,
						compact_total_runtimes,
						compact_total_lifetimes)) break;
				}

				steal_index++;

				// Get compaction queue length
				if (retval = work_progress.GetQueueLength(queue_index + 1, queue_length)) break;
				printf("%lld, %lld, ", (long long) iteration, (long long) queue_length);
				total_unculled += queue_length;

				// Reset compaction queue length
				if (retval = work_progress.SetQueueLength(queue_index + 1, 0)) break;

				if (util::B40CPerror(cudaBindTexture(
						0,
						local_cull::bitmask_tex_ref,
						(CollisionMask *) graph_slice->d_keep,
						channelDesc,
						bytes),
					"EnactorCull cudaBindTexture failed", __FILE__, __LINE__)) exit(1);

				// Compaction
				local_cull::Kernel<CompactPolicy>
					<<<compact_grid_size, CompactPolicy::THREADS>>>(
						iteration,
						queue_index,
						steal_index,
						graph_slice->frontier_queues.d_keys[selector ^ 1],		// d_in
						graph_slice->frontier_queues.d_keys[selector],			// d_out_row_offsets
						graph_slice->frontier_queues.d_values[selector],		// d_out_row_lengths
						(CollisionMask *) graph_slice->d_keep,
						graph_slice->d_row_offsets,
						graph_slice->d_source_path,
						this->work_progress);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "local_cull::Kernel failed ", __FILE__, __LINE__))) break;

				queue_index++;
				steal_index++;

				// Get compaction queue length
				if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
				printf("%lld, ", (long long) queue_length);


				if (!queue_length) {
					// Done
					break;
				}
			}

		} while(0);

		printf("\n");

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
		typename CsrProblem::SizeT 		src_offset,
		typename CsrProblem::SizeT 		src_length,
		int 							max_grid_size = 0)
	{
		if (this->cuda_props.device_sm_version >= 200) {

			//
			// Worker configs
			//

			// Expansion kernel config
			typedef neighbor_gather::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,
				false,					// BENCHMARK
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
				true, 					// DEQUEUE_PROBLEM_SIZE
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
				true,					// WORK_STEALING
				32,						// WARP_GATHER_THRESHOLD
				128 * 4, 				// CTA_GATHER_THRESHOLD,
				6> ExpandPolicy;

			// Compaction kernel config
			typedef local_cull::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,
				false,					// BENCHMARK
				INSTRUMENT, 			// INSTRUMENT
				true, 					// DEQUEUE_PROBLEM_SIZE
				8,						// CTA_OCCUPANCY
				7,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				2,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				9> CompactPolicy;

			//
			// Microbenchmark configs
			//

			// Compaction kernel config
			typedef local_cull::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,
				true,					// BENCHMARK
				INSTRUMENT, 			// INSTRUMENT
				true, 					// DEQUEUE_PROBLEM_SIZE
				8,						// CTA_OCCUPANCY
				7,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				2,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				9> BenchCompactPolicy;


			return EnactSearch<
				ExpandPolicy,
				CompactPolicy,
				BenchCompactPolicy,
				INSTRUMENT>(
					csr_problem,
					src,
					src_offset,
					src_length,
					max_grid_size);

		} else {
			printf("Not yet tuned for this architecture\n");
			return cudaErrorInvalidConfiguration;
		}
	}
    
};


} // namespace microbench
} // namespace bfs
} // namespace graph
} // namespace b40c
