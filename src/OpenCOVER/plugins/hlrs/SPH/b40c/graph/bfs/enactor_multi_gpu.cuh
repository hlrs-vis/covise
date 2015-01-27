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
 * Multi-GPU out-of-core BFS implementation (BFS level grid launch)
 ******************************************************************************/

#pragma once

#include <vector>

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/operators.cuh>

#include <b40c/graph/bfs/enactor_base.cuh>
#include <b40c/graph/bfs/problem_type.cuh>

#include <b40c/graph/bfs/compact_atomic/kernel.cuh>
#include <b40c/graph/bfs/compact_atomic/kernel_policy.cuh>

#include <b40c/graph/bfs/expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/expand_atomic/kernel_policy.cuh>

#include <b40c/graph/bfs/partition_compact/policy.cuh>
#include <b40c/graph/bfs/partition_compact/upsweep/kernel.cuh>
#include <b40c/graph/bfs/partition_compact/upsweep/kernel_policy.cuh>
#include <b40c/graph/bfs/partition_compact/downsweep/kernel.cuh>
#include <b40c/graph/bfs/partition_compact/downsweep/kernel_policy.cuh>

#include <b40c/graph/bfs/copy/kernel.cuh>
#include <b40c/graph/bfs/copy/kernel_policy.cuh>


namespace b40c {
namespace graph {
namespace bfs {



/**
 * Multi-GPU out-of-core BFS implementation (BFS level grid launch)
 *  
 * Each iteration is performed by its own kernel-launch.
 *
 * All GPUs must be of the same SM architectural version (e.g., SM2.0).
 */
class EnactorMultiGpu : public EnactorBase
{
public :

	//---------------------------------------------------------------------
	// Policy Structures
	//---------------------------------------------------------------------

	template <bool INSTRUMENT, typename CsrProblem, int SM_ARCH>
	struct Policy;

	/**
	 * SM2.0 policy
	 */
	template <bool INSTRUMENT, typename CsrProblem>
	struct Policy<INSTRUMENT, CsrProblem, 200>
	{
		// Compaction kernel config
		typedef compact_atomic::KernelPolicy<
			typename CsrProblem::ProblemType,
			200,
			INSTRUMENT, 			// INSTRUMENT
			false, 					// DEQUEUE_PROBLEM_SIZE
			8,						// CTA_OCCUPANCY
			7,						// LOG_THREADS
			0,						// LOG_LOAD_VEC_SIZE
			2,						// LOG_LOADS_PER_TILE
			5,						// LOG_RAKING_THREADS
			util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
			util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
			false,					// WORK_STEALING
			9> CompactPolicy;		// LOG_SCHEDULE_GRANULARITY

		// Expansion kernel config
		typedef expand_atomic::KernelPolicy<
			typename CsrProblem::ProblemType,
			200,
			INSTRUMENT, 			// INSTRUMENT
			0, 						// SATURATION_QUIT
			8,						// CTA_OCCUPANCY
			7,						// LOG_THREADS
			0,						// LOG_LOAD_VEC_SIZE
			0,						// LOG_LOADS_PER_TILE
			5,						// LOG_RAKING_THREADS
			util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
			util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
			util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
			util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
			util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
			true,					// WORK_STEALING
			32,						// WARP_GATHER_THRESHOLD
			128 * 4, 				// CTA_GATHER_THRESHOLD,
			6> ExpandPolicy;		// LOG_SCHEDULE_GRANULARITY


		// Make sure we satisfy the tuning constraints in partition::[up|down]sweep::tuning_policy.cuh
		typedef partition_compact::Policy<
			// Problem Type
			typename CsrProblem::ProblemType,
			200,
			INSTRUMENT, 			// INSTRUMENT
			CsrProblem::LOG_MAX_GPUS,	// LOG_BINS
			9,						// LOG_SCHEDULE_GRANULARITY
			util::io::ld::NONE,		// CACHE_MODIFIER
			util::io::st::NONE,		// CACHE_MODIFIER

			8,						// UPSWEEP_CTA_OCCUPANCY
			7,						// UPSWEEP_LOG_THREADS
			0,						// UPSWEEP_LOG_LOAD_VEC_SIZE
			2,						// UPSWEEP_LOG_LOADS_PER_TILE

			7,						// SPINE_LOG_THREADS
			2,						// SPINE_LOG_LOAD_VEC_SIZE
			0,						// SPINE_LOG_LOADS_PER_TILE
			5,						// SPINE_LOG_RAKING_THREADS

			partition::downsweep::SCATTER_DIRECT,		// DOWNSWEEP_SCATTER_STRATEGY
			8,						// DOWNSWEEP_CTA_OCCUPANCY
			7,						// DOWNSWEEP_LOG_THREADS
			1,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
			1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
			0,						// DOWNSWEEP_LOG_CYCLES_PER_TILE
			6> PartitionPolicy;		// DOWNSWEEP_LOG_RAKING_THREADS

		// Copy kernel config
		typedef copy::KernelPolicy<
			typename CsrProblem::ProblemType,
			200,
			INSTRUMENT, 			// INSTRUMENT
			false, 					// DEQUEUE_PROBLEM_SIZE
			6,						// LOG_SCHEDULE_GRANULARITY
			8,						// CTA_OCCUPANCY
			6,						// LOG_THREADS
			0,						// LOG_LOAD_VEC_SIZE
			0,						// LOG_LOADS_PER_TILE
			util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
			util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
			false> CopyPolicy;		// WORK_STEALING
	};


protected:

	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Management structure for each GPU
	 */
	struct GpuControlBlock
	{
		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		bool DEBUG;

		// GPU index
		int gpu;

		// GPU cuda properties
		util::CudaProperties cuda_props;

		// Queue size counters and accompanying functionality
		util::CtaWorkProgressLifetime work_progress;

		// Partitioning spine storage
		util::Spine spine;
		int spine_elements;

		int compact_grid_size;			// Compaction grid size
		int expand_grid_size;			// Expansion grid size
		int partition_grid_size;		// Partition/compact grid size
		int copy_grid_size;				// Copy grid size

		long long iteration;			// BFS iteration
		long long queue_index;			// Queuing index
		long long steal_index;			// Work stealing index
		long long queue_length;			// Current queue size
		int selector;					// Ping-pong storage selector

		// Kernel duty stats
		util::KernelRuntimeStatsLifetime compact_kernel_stats;
		util::KernelRuntimeStatsLifetime expand_kernel_stats;
		util::KernelRuntimeStatsLifetime partition_kernel_stats;
		util::KernelRuntimeStatsLifetime copy_kernel_stats;


		//---------------------------------------------------------------------
		// Methods
		//---------------------------------------------------------------------

		/**
		 * Constructor
		 */
		GpuControlBlock(int gpu, bool DEBUG = false) :
			gpu(gpu),
			DEBUG(DEBUG),
			cuda_props(gpu),
			spine(true),				// Host-mapped spine
			spine_elements(0),
			compact_grid_size(0),
			expand_grid_size(0),
			partition_grid_size(0),
			copy_grid_size(0),
			iteration(0),
			selector(0),
			steal_index(0),
			queue_index(0),
			queue_length(0)
		{}


		/**
		 * Returns the default maximum number of threadblocks that should be
		 * launched for this GPU.
		 */
		int MaxGridSize(int cta_occupancy, int max_grid_size)
		{
			if (max_grid_size <= 0) {
				// No override: Fully populate all SMs
				max_grid_size = cuda_props.device_props.multiProcessorCount * cta_occupancy;
			}

			return max_grid_size;
		}


		/**
		 * Setup / lazy initialization
		 */
	    template <
	    	typename CompactPolicy,
	    	typename ExpandPolicy,
	    	typename PartitionPolicy,
	    	typename CopyPolicy>
		cudaError_t Setup(int max_grid_size, int num_gpus)
		{
	    	cudaError_t retval = cudaSuccess;

			do {
		    	// Determine grid size(s)
				int compact_min_occupancy 		= CompactPolicy::CTA_OCCUPANCY;
				compact_grid_size 				= MaxGridSize(compact_min_occupancy, max_grid_size);

				int expand_min_occupancy 		= ExpandPolicy::CTA_OCCUPANCY;
				expand_grid_size 				= MaxGridSize(expand_min_occupancy, max_grid_size);

				int partition_min_occupancy		= B40C_MIN((int) PartitionPolicy::Upsweep::MAX_CTA_OCCUPANCY, (int) PartitionPolicy::Downsweep::MAX_CTA_OCCUPANCY);
				partition_grid_size 			= MaxGridSize(partition_min_occupancy, max_grid_size);

				int copy_min_occupancy			= CopyPolicy::CTA_OCCUPANCY;
				copy_grid_size 					= MaxGridSize(copy_min_occupancy, max_grid_size);

				// Setup partitioning spine
				spine_elements = (partition_grid_size * PartitionPolicy::Upsweep::BINS) + 1;
				if (retval = spine.template Setup<typename PartitionPolicy::SizeT>(spine_elements)) break;

				if (DEBUG) printf("Gpu %d compact  min occupancy %d, grid size %d\n",
					gpu, compact_min_occupancy, compact_grid_size);
				if (DEBUG) printf("Gpu %d expand min occupancy %d, grid size %d\n",
					gpu, expand_min_occupancy, expand_grid_size);
				if (DEBUG) printf("Gpu %d partition min occupancy %d, grid size %d, spine elements %d\n",
					gpu, partition_min_occupancy, partition_grid_size, spine_elements);
				if (DEBUG) printf("Gpu %d copy min occupancy %d, grid size %d\n",
					gpu, copy_min_occupancy, copy_grid_size);

				// Setup work progress
				if (retval = work_progress.Setup()) break;

			} while (0);

			// Reset statistics
			iteration = 0;
			selector = 0;
			queue_index = 0;
			steal_index = 0;
			queue_length = 0;

			return retval;
		}


	    /**
	     * Updates queue length from work progress
	     *
	     * (SizeT may be different for each graph search)
	     */
		template <typename SizeT>
	    cudaError_t UpdateQueueLength()
	    {
	    	SizeT length;
	    	cudaError_t retval = work_progress.GetQueueLength(queue_index, length);
	    	queue_length = length;

	    	return retval;
	    }
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Vector of GpuControlBlocks (one for each GPU)
	std::vector <GpuControlBlock *> control_blocks;

	bool DEBUG2;

	//---------------------------------------------------------------------
	// Utility Methods
	//---------------------------------------------------------------------


public: 	
	
	/**
	 * Constructor
	 */
	EnactorMultiGpu(bool DEBUG = false) :
		EnactorBase(DEBUG),
		DEBUG2(false)
	{}


	/**
	 * Resets control blocks
	 */
	void ResetControlBlocks()
	{
		// Cleanup control blocks on the heap
		for (typename std::vector<GpuControlBlock*>::iterator itr = control_blocks.begin();
			itr != control_blocks.end();
			itr++)
		{
			if (*itr) delete (*itr);
		}

		control_blocks.clear();
	}


	/**
	 * Destructor
	 */
	virtual ~EnactorMultiGpu()
	{
		ResetControlBlocks();
	}


    /**
     * Obtain statistics about the last BFS search enacted 
     */
	template <typename VertexId>
    void GetStatistics(
    	long long &total_queued,
    	VertexId &search_depth,
    	double &avg_live)
    {
		// TODO
    	total_queued = 0;
    	search_depth = 0;
    	avg_live = 0;
    }


	/**
	 * Search setup / lazy initialization
	 */
    template <
    	typename CompactPolicy,
    	typename ExpandPolicy,
    	typename PartitionPolicy,
    	typename CopyPolicy,
    	typename CsrProblem>
	cudaError_t Setup(
		CsrProblem 		&csr_problem,
		int 			max_grid_size)
    {
		typedef typename CsrProblem::SizeT 			SizeT;
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::CollisionMask 	CollisionMask;

		cudaError_t retval = cudaSuccess;

    	do {
			// Check if last run was with an different number of GPUs (in which
			// case the control blocks are all misconfigured)
			if (control_blocks.size() != csr_problem.num_gpus) {

				ResetControlBlocks();

				for (int i = 0; i < csr_problem.num_gpus; i++) {

					// Set device
					if (retval = util::B40CPerror(cudaSetDevice(csr_problem.graph_slices[i]->gpu),
						"EnactorMultiGpu cudaSetDevice failed", __FILE__, __LINE__)) break;

					control_blocks.push_back(
						new GpuControlBlock(csr_problem.graph_slices[i]->gpu,
						DEBUG));
				}
			}

			// Setup control blocks
			for (int i = 0; i < csr_problem.num_gpus; i++) {

				// Set device
				if (retval = util::B40CPerror(cudaSetDevice(csr_problem.graph_slices[i]->gpu),
					"EnactorMultiGpu cudaSetDevice failed", __FILE__, __LINE__)) break;

				if (retval = control_blocks[i]->template Setup<CompactPolicy, ExpandPolicy, PartitionPolicy, CopyPolicy>(
					max_grid_size, csr_problem.num_gpus)) break;

				// Bind bitmask textures
				int bytes = (csr_problem.nodes + 8 - 1) / 8;
				cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
				if (retval = util::B40CPerror(cudaBindTexture(
						0,
						compact_atomic::BitmaskTex<CollisionMask>::ref,
						csr_problem.graph_slices[i]->d_collision_cache,
						bitmask_desc,
						bytes),
					"EnactorMultiGpu cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

				// Bind row-offsets texture
				cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
				if (retval = util::B40CPerror(cudaBindTexture(
						0,
						expand_atomic::RowOffsetTex<SizeT>::ref,
						csr_problem.graph_slices[i]->d_row_offsets,
						row_offsets_desc,
						(csr_problem.graph_slices[i]->nodes + 1) * sizeof(SizeT)),
					"EnactorMultiGpu cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

			}
			if (retval) break;

    	} while (0);

    	return retval;
    }



	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <
    	typename CompactPolicy,
    	typename ExpandPolicy,
    	typename PartitionPolicy,
    	typename CopyPolicy,
    	bool INSTRUMENT,
    	typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 							&csr_problem,
		typename CsrProblem::VertexId 		src,
		int 								max_grid_size = 0)
	{
		typedef typename CsrProblem::VertexId			VertexId;
		typedef typename CsrProblem::SizeT				SizeT;
		typedef typename CsrProblem::GraphSlice			GraphSlice;

		typedef typename PartitionPolicy::Upsweep		PartitionUpsweep;
		typedef typename PartitionPolicy::Spine			PartitionSpine;
		typedef typename PartitionPolicy::Downsweep		PartitionDownsweep;

		cudaError_t retval = cudaSuccess;
		bool done;

		do {

			// Number of partitioning bins per GPU (in case we over-partition)
			int bins_per_gpu = (csr_problem.num_gpus == 1) ?
				PartitionPolicy::Upsweep::BINS :
				1;
			printf("Partition bins per GPU: %d\n", bins_per_gpu);

			// Search setup / lazy initialization
			if (retval = Setup<CompactPolicy, ExpandPolicy, PartitionPolicy, CopyPolicy>(
				csr_problem, max_grid_size)) break;

			// Mask in owner gpu of source;
			int src_owner = csr_problem.GpuIndex(src);
			src |= (src_owner << CsrProblem::ProblemType::GPU_MASK_SHIFT);


			//---------------------------------------------------------------------
			// Compact work queues (first iteration)
			//---------------------------------------------------------------------

			for (int i = 0; i < csr_problem.num_gpus; i++) {

				GpuControlBlock *control 	= control_blocks[i];
				GraphSlice *slice 			= csr_problem.graph_slices[i];

				// Set device
				if (retval = util::B40CPerror(cudaSetDevice(control->gpu),
					"EnactorMultiGpu cudaSetDevice failed", __FILE__, __LINE__)) break;

				bool owns_source = (control->gpu == src_owner);
				if (owns_source) {
					printf("GPU %d owns source 0x%X\n", control->gpu, src);
				}

				// Compaction
				compact_atomic::Kernel<CompactPolicy>
						<<<control->compact_grid_size, CompactPolicy::THREADS, 0, slice->stream>>>(
					(owns_source) ? src : -1,
					control->iteration,
					(owns_source) ? 1 : 0,																			//
					control->queue_index,
					control->steal_index,
					csr_problem.num_gpus,
					NULL,																		// d_done (not used)
					slice->frontier_queues.d_keys[control->selector ^ 1],						// in vertices
					slice->d_multigpu_vqueue,													// out vertices
					(VertexId *) slice->frontier_queues.d_values[control->selector ^ 1],		// in parents
					slice->d_source_path,
					slice->d_collision_cache,
					control->work_progress,
					control->expand_kernel_stats);

				if (DEBUG && (retval = util::B40CPerror(cudaDeviceSynchronize(),
					"EnactorMultiGpu expand_atomic::Kernel failed", __FILE__, __LINE__))) break;

				control->queue_index++;
				control->steal_index++;
			}
			if (retval) break;

			// BFS passes
			while (true) {

				//---------------------------------------------------------------------
				// Expand work queues
				//---------------------------------------------------------------------

				for (int i = 0; i < csr_problem.num_gpus; i++) {

					GpuControlBlock *control 	= control_blocks[i];
					GraphSlice *slice 			= csr_problem.graph_slices[i];

					// Set device
					if (retval = util::B40CPerror(cudaSetDevice(control->gpu),
						"EnactorMultiGpu cudaSetDevice failed", __FILE__, __LINE__)) break;

					expand_atomic::Kernel<ExpandPolicy>
							<<<control->expand_grid_size, ExpandPolicy::THREADS, 0, slice->stream>>>(
						control->queue_index,
						control->steal_index,
						csr_problem.num_gpus,
						NULL,														// d_done (not used)
						slice->d_multigpu_vqueue,									// in vertices
						slice->frontier_queues.d_keys[control->selector],			// out vertices
						slice->frontier_queues.d_values[control->selector],			// out parents
						slice->d_column_indices,
						slice->d_row_offsets,
						control->work_progress,
						control->expand_kernel_stats);

					if (DEBUG && (retval = util::B40CPerror(cudaDeviceSynchronize(),
						"EnactorMultiGpu expand_atomic::Kernel failed", __FILE__, __LINE__))) break;

					control->queue_index++;
					control->steal_index++;
					control->iteration++;
				}
				if (retval) break;


				//---------------------------------------------------------------------
				// Partition/compact work queues
				//---------------------------------------------------------------------

				for (int i = 0; i < csr_problem.num_gpus; i++) {

					GpuControlBlock *control 	= control_blocks[i];
					GraphSlice *slice 			= csr_problem.graph_slices[i];

					// Set device
					if (retval = util::B40CPerror(cudaSetDevice(control->gpu),
						"EnactorMultiGpu cudaSetDevice failed", __FILE__, __LINE__)) break;

					// Upsweep
					partition_compact::upsweep::Kernel<PartitionUpsweep>
							<<<control->partition_grid_size, PartitionUpsweep::THREADS, 0, slice->stream>>>(
						control->queue_index,
						csr_problem.num_gpus,
						slice->frontier_queues.d_keys[control->selector],			// in vertices
						slice->d_keep,
						(SizeT *) control->spine.d_spine,
						slice->d_collision_cache,
						control->work_progress,
						control->partition_kernel_stats);

					if (DEBUG && (retval = util::B40CPerror(cudaDeviceSynchronize(),
						"EnactorMultiGpu partition_compact::upsweep::Kernel failed", __FILE__, __LINE__))) break;

					if (DEBUG2) {
						printf("Presorted spine on gpu %d (%lld elements):\n",
							control->gpu,
							(long long) control->spine_elements);
						DisplayDeviceResults(
							(SizeT *) control->spine.d_spine,
							control->spine_elements);
					}

					// Spine
					PartitionPolicy::SpineKernel()<<<1, PartitionSpine::THREADS, 0, slice->stream>>>(
						(SizeT*) control->spine.d_spine,
						(SizeT*) control->spine.d_spine,
						control->spine_elements);

					if (DEBUG && (retval = util::B40CPerror(cudaDeviceSynchronize(),
						"EnactorMultiGpu SpineKernel failed", __FILE__, __LINE__))) break;

					if (DEBUG2) {
						printf("Postsorted spine on gpu %d (%lld elements):\n",
							control->gpu,
							(long long) control->spine_elements);
						DisplayDeviceResults(
							(SizeT *) control->spine.d_spine,
							control->spine_elements);
					}

					// Downsweep
					partition_compact::downsweep::Kernel<PartitionDownsweep>
							<<<control->partition_grid_size, PartitionDownsweep::THREADS, 0, slice->stream>>>(
						control->queue_index,
						csr_problem.num_gpus,
						slice->frontier_queues.d_keys[control->selector],						// in vertices
						slice->frontier_queues.d_keys[control->selector ^ 1],					// out vertices
						(VertexId *) slice->frontier_queues.d_values[control->selector],		// in parents
						(VertexId *) slice->frontier_queues.d_values[control->selector ^ 1],	// out parents
						slice->d_keep,
						(SizeT *) control->spine.d_spine,
						control->work_progress,
						control->partition_kernel_stats);

					if (DEBUG && (retval = util::B40CPerror(cudaDeviceSynchronize(),
						"EnactorMultiGpu DownsweepKernel failed", __FILE__, __LINE__))) break;

					control->queue_index++;
				}
				if (retval) break;

				//---------------------------------------------------------------------
				// Synchronization point (to make spines coherent)
				//---------------------------------------------------------------------

				done = true;
				for (int i = 0; i < csr_problem.num_gpus; i++) {

					GpuControlBlock *control 	= control_blocks[i];
					GraphSlice *slice 			= csr_problem.graph_slices[i];

					if (retval = util::B40CPerror(cudaSetDevice(control->gpu),
						"EnactorMultiGpu cudaSetDevice failed", __FILE__, __LINE__)) break;

					// The memcopy for spine sync synchronizes this GPU
					control->spine.Sync();

					SizeT *spine = (SizeT *) control->spine.h_spine;
					if (spine[control->spine_elements - 1]) done = false;

					if (DEBUG2) {
						printf("Iteration %lld sort-compacted queue on gpu %d (%lld elements):\n",
							(long long) control->iteration,
							control->gpu,
							(long long) spine[control->spine_elements - 1]);
						DisplayDeviceResults(
							slice->frontier_queues.d_keys[control->selector],
							spine[control->spine_elements - 1]);
						printf("Source distance vector on gpu %d:\n", control->gpu);
						DisplayDeviceResults(
							slice->d_source_path,
							slice->nodes);
					}
				}
				if (retval) break;

				// Check if all done in all GPUs
				if (done) break;

				if (DEBUG2) printf("---------------------------------------------------------\n");


				//---------------------------------------------------------------------
				// Stream-compact work queues
				//---------------------------------------------------------------------

				for (int i = 0; i < csr_problem.num_gpus; i++) {

					GpuControlBlock *control 	= control_blocks[i];
					GraphSlice *slice 			= csr_problem.graph_slices[i];

					// Set device
					if (retval = util::B40CPerror(cudaSetDevice(control->gpu),
						"EnactorMultiGpu cudaSetDevice failed", __FILE__, __LINE__)) break;

					// Stream in and filter inputs from all gpus (including ourselves)
					for (int j = 0; j < csr_problem.num_gpus; j++) {

						// Starting with ourselves (must copy our own bin first), stream
						// bins into our queue
						int peer 							= (i + j) % csr_problem.num_gpus;
						GpuControlBlock *peer_control 		= control_blocks[peer];
						GraphSlice *peer_slice 				= csr_problem.graph_slices[peer];
						SizeT *peer_spine 			= (SizeT*) peer_control->spine.h_spine;

						SizeT queue_offset 	= peer_spine[bins_per_gpu * i * peer_control->partition_grid_size];
						SizeT queue_oob 	= peer_spine[bins_per_gpu * (i + 1) * peer_control->partition_grid_size];
						SizeT num_elements	= queue_oob - queue_offset;

						if (DEBUG2) {
							printf("Gpu %d getting %d from gpu %d selector %d, queue_offset: %d @ %d, queue_oob: %d @ %d\n",
								i,
								num_elements,
								peer,
								peer_control->selector,
								queue_offset,
								bins_per_gpu * i * peer_control->partition_grid_size,
								queue_oob,
								bins_per_gpu * (i + 1) * peer_control->partition_grid_size);
							fflush(stdout);
						}

						if (slice->gpu == peer_slice->gpu) {

							util::CtaWorkDistribution<SizeT> work_decomposition;
							work_decomposition.template Init<CopyPolicy::LOG_SCHEDULE_GRANULARITY>(
								num_elements, control->copy_grid_size);

							// Simply copy
							copy::Kernel<CopyPolicy>
								<<<control->copy_grid_size, CopyPolicy::THREADS, 0, slice->stream>>>(
									control->iteration,
									num_elements,
									control->queue_index,
									control->steal_index,
									csr_problem.num_gpus,
									peer_slice->frontier_queues.d_keys[control->selector ^ 1] + queue_offset,					// in vertices
									slice->d_multigpu_vqueue,																	// out vertices
									(VertexId *) peer_slice->frontier_queues.d_values[control->selector] + queue_offset,		// in parents
									slice->d_source_path,
									control->work_progress,
									control->copy_kernel_stats);

							if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(),
								"EnactorMultiGpu copy::Kernel failed ", __FILE__, __LINE__))) break;

						} else {

							// Compaction
							compact_atomic::Kernel<CompactPolicy>
								<<<control->compact_grid_size, CompactPolicy::THREADS, 0, slice->stream>>>(
									-1,														// source (not used)
									control->iteration,
									num_elements,
									control->queue_index,
									control->steal_index,
									csr_problem.num_gpus,
									NULL,																						// d_done (not used)
									peer_slice->frontier_queues.d_keys[control->selector ^ 1] + queue_offset,					// in vertices
									slice->d_multigpu_vqueue,																	// out vertices
									(VertexId *) peer_slice->frontier_queues.d_values[control->selector ^ 1] + queue_offset,		// in parents
									slice->d_source_path,
									slice->d_collision_cache,
									control->work_progress,
									control->expand_kernel_stats);
							if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(),
								"EnactorMultiGpu compact_atomic::Kernel failed ", __FILE__, __LINE__))) break;

						}
						control->steal_index++;
					}

					control->queue_index++;
				}
				if (retval) break;
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
		CsrProblem 							&csr_problem,
		typename CsrProblem::VertexId 		src,
		int 								max_grid_size = 0)
	{
		typedef typename CsrProblem::VertexId			VertexId;
		typedef typename CsrProblem::SizeT				SizeT;

		if (this->cuda_props.device_sm_version >= 200) {

			typedef Policy<INSTRUMENT, CsrProblem, 200> CsrPolicy;

			return EnactSearch<
				typename CsrPolicy::CompactPolicy,
				typename CsrPolicy::ExpandPolicy,
				typename CsrPolicy::PartitionPolicy,
				typename CsrPolicy::CopyPolicy,
				INSTRUMENT>(csr_problem, src, max_grid_size);

		} else {
			printf("Not yet tuned for this architecture\n");
			return cudaErrorInvalidConfiguration;
		}
	}

    
};



} // namespace bfs
} // namespace graph
} // namespace b40c
