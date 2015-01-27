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
 * BFS atomic expand-compact kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/expand_compact_atomic/cta.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace expand_compact_atomic {


/**
 * Sweep compact-expand pass (non-workstealing)
 */
template <typename KernelPolicy, bool WORK_STEALING>
struct SweepPass
{
	template <typename SmemStorage>
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::VertexId 		&iteration,
		typename KernelPolicy::VertexId 		&queue_index,
		typename KernelPolicy::VertexId 		&steal_index,
		typename KernelPolicy::VertexId 		*&d_in,
		typename KernelPolicy::VertexId 		*&d_out,
		typename KernelPolicy::VertexId 		*&d_parent_in,
		typename KernelPolicy::VertexId 		*&d_parent_out,
		typename KernelPolicy::VertexId			*&d_column_indices,
		typename KernelPolicy::SizeT			*&d_row_offsets,
		typename KernelPolicy::VertexId			*&d_source_path,
		typename KernelPolicy::CollisionMask 	*&d_collision_cache,
		util::CtaWorkProgress 					&work_progress,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
		SmemStorage								&smem_storage)
	{
		typedef Cta<KernelPolicy> 						Cta;
		typedef typename KernelPolicy::SizeT 			SizeT;

		// Determine our threadblock's work range
		util::CtaWorkLimits<SizeT> work_limits;
		work_decomposition.template GetCtaWorkLimits<
			KernelPolicy::LOG_TILE_ELEMENTS,
			KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

		// Return if we have no work to do
		if (!work_limits.elements) {
			return;
		}

		// CTA processing abstraction
		Cta cta(
			iteration,
			queue_index,
			smem_storage,
			d_in,
			d_out,
			d_parent_in,
			d_parent_out,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			d_collision_cache,
			work_progress);

		// Process full tiles
		while (work_limits.offset < work_limits.guarded_offset) {

			cta.ProcessTile(work_limits.offset);
			work_limits.offset += KernelPolicy::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-i/o
		if (work_limits.guarded_elements) {
			cta.ProcessTile(
				work_limits.offset,
				work_limits.guarded_elements);
		}
	}
};


template <typename SizeT, typename StealIndex>
__device__ __forceinline__ SizeT StealWork(
	util::CtaWorkProgress &work_progress,
	int count,
	StealIndex steal_index)
{
	__shared__ SizeT s_offset;		// The offset at which this CTA performs tile processing, shared by all

	// Thread zero atomically steals work from the progress counter
	if (threadIdx.x == 0) {
		s_offset = work_progress.Steal<SizeT>(count, steal_index);
	}

	__syncthreads();		// Protect offset

	return s_offset;
}



/**
 * Sweep compact-expand pass (workstealing)
 */
template <typename KernelPolicy>
struct SweepPass <KernelPolicy, true>
{
	template <typename SmemStorage>
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::VertexId 		&iteration,
		typename KernelPolicy::VertexId 		&queue_index,
		typename KernelPolicy::VertexId 		&steal_index,
		typename KernelPolicy::VertexId 		*&d_in,
		typename KernelPolicy::VertexId 		*&d_out,
		typename KernelPolicy::VertexId 		*&d_parent_in,
		typename KernelPolicy::VertexId 		*&d_parent_out,
		typename KernelPolicy::VertexId			*&d_column_indices,
		typename KernelPolicy::SizeT			*&d_row_offsets,
		typename KernelPolicy::VertexId			*&d_source_path,
		typename KernelPolicy::CollisionMask 	*&d_collision_cache,
		util::CtaWorkProgress 					&work_progress,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
		SmemStorage								&smem_storage)
	{
		typedef Cta<KernelPolicy> 					Cta;
		typedef typename KernelPolicy::SizeT 		SizeT;

		// CTA processing abstraction
		Cta cta(
			iteration,
			queue_index,
			smem_storage,
			d_in,
			d_out,
			d_parent_in,
			d_parent_out,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			d_collision_cache,
			work_progress);

		// Total number of elements in full tiles
		SizeT unguarded_elements = work_decomposition.num_elements & (~(KernelPolicy::TILE_ELEMENTS - 1));

		// Worksteal full tiles, if any
		SizeT offset;
		while ((offset = StealWork<SizeT>(work_progress, KernelPolicy::TILE_ELEMENTS, steal_index)) < unguarded_elements) {
			cta.ProcessTile(offset);
		}

		// Last CTA does any extra, guarded work (first tile seen)
		if (blockIdx.x == gridDim.x - 1) {
			SizeT guarded_elements = work_decomposition.num_elements - unguarded_elements;
			cta.ProcessTile(unguarded_elements, guarded_elements);
		}
	}
};


/******************************************************************************
 * Sweep Kernel Entrypoint
 ******************************************************************************/

/**
 * Kernel interface for a single-launch expand-contract BFS traversal
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void KernelGlobalBarrier(
	typename KernelPolicy::VertexId			iteration,
	typename KernelPolicy::VertexId			queue_index,
	typename KernelPolicy::VertexId			steal_index,
	typename KernelPolicy::VertexId 		src,
	typename KernelPolicy::VertexId 		*d_in,
	typename KernelPolicy::VertexId 		*d_out,
	typename KernelPolicy::VertexId 		*d_parent_in,
	typename KernelPolicy::VertexId 		*d_parent_out,

	typename KernelPolicy::VertexId			*d_column_indices,
	typename KernelPolicy::SizeT			*d_row_offsets,
	typename KernelPolicy::VertexId			*d_source_path,
	typename KernelPolicy::CollisionMask 	*d_collision_cache,
	util::CtaWorkProgress 					work_progress,
	util::GlobalBarrier						global_barrier,

	util::KernelRuntimeStats				kernel_stats,
	typename KernelPolicy::VertexId			*d_iteration)
{
	typedef typename KernelPolicy::VertexId VertexId;
	typedef typename KernelPolicy::SizeT SizeT;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStart();
	}

	if (iteration == 0) {

		if (threadIdx.x < util::CtaWorkProgress::COUNTERS) {

			// Reset all counters
			work_progress.template Reset<SizeT>();

			// Determine work decomposition for first iteration
			if (threadIdx.x == 0) {

				// We'll be the only block with active work this iteration.
				// Enqueue the source for us to subsequently process.
				util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(src, d_in);

				if (KernelPolicy::MARK_PARENTS) {
					// Enqueue parent of source
					typename KernelPolicy::VertexId parent = -2;
					util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(parent, d_parent_in);
				}

				// Update source path with current iteration (mooch parent)
				util::io::ModifiedStore<util::io::st::cg>::St(
					iteration,
					d_source_path + src);

				// Initialize work decomposition in smem
				SizeT num_elements = 1;
				smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
					num_elements, gridDim.x);
			}
		}

	} else {

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
			if (KernelPolicy::INSTRUMENT && (blockIdx.x == 0)) {
				kernel_stats.Aggregate(num_elements);
			}

			// Initialize work decomposition in smem
			smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);

		}
	}

	// Barrier to protect work decomposition
	__syncthreads();

	// Don't do workstealing this iteration because without a
	// global barrier after queue-reset, the queue may be inconsistent
	// across CTAs
	SweepPass<KernelPolicy, false>::Invoke(
		iteration,
		queue_index,
		steal_index,
		d_in,
		d_out,
		d_parent_in,
		d_parent_out,
		d_column_indices,
		d_row_offsets,
		d_source_path,
		d_collision_cache,
		work_progress,
		smem_storage.state.work_decomposition,
		smem_storage);

	iteration++;
	queue_index++;
	steal_index++;

	if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStop();
	}

	global_barrier.Sync();

	while (true) {

		//---------------------------------------------------------------------
		// Flip
		//---------------------------------------------------------------------

		if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStart();
		}

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
			if (KernelPolicy::INSTRUMENT && (blockIdx.x == 0)) {
				kernel_stats.Aggregate(num_elements);
			}

			// Initialize work decomposition in smem
			smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);

		}

		// Barrier to protect work decomposition
		__syncthreads();

		if ((!smem_storage.state.work_decomposition.num_elements) ||
			(KernelPolicy::SATURATION_QUIT && (smem_storage.state.work_decomposition.num_elements > gridDim.x * KernelPolicy::TILE_ELEMENTS * KernelPolicy::SATURATION_QUIT)))
		{
			break;
		}

		SweepPass<KernelPolicy, KernelPolicy::WORK_STEALING>::Invoke(
			iteration,
			queue_index,
			steal_index,
			d_out,
			d_in,
			d_parent_out,
			d_parent_in,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			d_collision_cache,
			work_progress,
			smem_storage.state.work_decomposition,
			smem_storage);

		iteration++;
		queue_index++;
		steal_index++;

		if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStop();
		}

		global_barrier.Sync();

		//---------------------------------------------------------------------
		// Flop
		//---------------------------------------------------------------------

		if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStart();
		}

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
			if (KernelPolicy::INSTRUMENT && (blockIdx.x == 0)) {
				kernel_stats.Aggregate(num_elements);
			}

			// Initialize work decomposition in smem
			smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);
		}

		// Barrier to protect work decomposition
		__syncthreads();

		// Check if done
		if ((!smem_storage.state.work_decomposition.num_elements) ||
			(KernelPolicy::SATURATION_QUIT && (smem_storage.state.work_decomposition.num_elements > gridDim.x * KernelPolicy::TILE_ELEMENTS * KernelPolicy::SATURATION_QUIT)))
		{
			break;
		}

		SweepPass<KernelPolicy, KernelPolicy::WORK_STEALING>::Invoke(
			iteration,
			queue_index,
			steal_index,
			d_in,
			d_out,
			d_parent_in,
			d_parent_out,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			d_collision_cache,
			work_progress,
			smem_storage.state.work_decomposition,
			smem_storage);

		iteration++;
		queue_index++;
		steal_index++;

		if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStop();
		}

		global_barrier.Sync();
	}

	// Write out our final iteration
	if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
		d_iteration[0] = iteration;
	}

	if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStop();
		kernel_stats.Flush();
	}
}


/**
 * Kernel interface for a single expand-contract BFS iteration
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::VertexId			iteration,
	typename KernelPolicy::VertexId			queue_index,
	typename KernelPolicy::VertexId			steal_index,
	volatile int							*d_done,
	typename KernelPolicy::VertexId 		src,
	typename KernelPolicy::VertexId 		*d_in,
	typename KernelPolicy::VertexId 		*d_out,
	typename KernelPolicy::VertexId 		*d_parent_in,
	typename KernelPolicy::VertexId 		*d_parent_out,
	typename KernelPolicy::VertexId			*d_column_indices,
	typename KernelPolicy::SizeT			*d_row_offsets,
	typename KernelPolicy::VertexId			*d_source_path,
	typename KernelPolicy::CollisionMask 	*d_collision_cache,
	util::CtaWorkProgress 					work_progress,
	util::KernelRuntimeStats				kernel_stats)
{
	typedef typename KernelPolicy::VertexId VertexId;
	typedef typename KernelPolicy::SizeT SizeT;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStart();
	}

	if (iteration == 0) {

		if (threadIdx.x < util::CtaWorkProgress::COUNTERS) {

			// Reset all counters
			work_progress.template Reset<SizeT>();

			// Determine work decomposition for first iteration
			if (threadIdx.x == 0) {

				// We'll be the only block with active work this iteration.
				// Enqueue the source for us to subsequently process.
				util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(src, d_in);

				if (KernelPolicy::MARK_PARENTS) {
					// Enqueue parent of source
					typename KernelPolicy::VertexId parent = -2;
					util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(parent, d_parent_in);
				}

				// Update source path with current iteration (mooch parent)
				util::io::ModifiedStore<util::io::st::cg>::St(
					iteration,
					d_source_path + src);

				// Initialize work decomposition in smem
				SizeT num_elements = 1;
				smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
					num_elements, gridDim.x);
			}
		}

		// Barrier to protect work decomposition
		__syncthreads();

		// Don't do workstealing this iteration because without a
		// global barrier after queue-reset, the queue may be inconsistent
		// across CTAs
		SweepPass<KernelPolicy, false>::Invoke(
			iteration,
			queue_index,
			steal_index,
			d_in,
			d_out,
			d_parent_in,
			d_parent_out,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			d_collision_cache,
			work_progress,
			smem_storage.state.work_decomposition,
			smem_storage);

	} else {

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
			if (KernelPolicy::INSTRUMENT && (blockIdx.x == 0)) {
				kernel_stats.Aggregate(num_elements);
			}

			// Signal to host that we're done
			if ((num_elements == 0) ||
				(KernelPolicy::SATURATION_QUIT && (num_elements > gridDim.x * KernelPolicy::TILE_ELEMENTS * KernelPolicy::SATURATION_QUIT)))
			{
				if (d_done) d_done[0] = 1;
			}

			// Initialize work decomposition in smem
			smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);
		}

		// Barrier to protect work decomposition
		__syncthreads();

		SweepPass<KernelPolicy, KernelPolicy::WORK_STEALING>::Invoke(
			iteration,
			queue_index,
			steal_index,
			d_in,
			d_out,
			d_parent_in,
			d_parent_out,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			d_collision_cache,
			work_progress,
			smem_storage.state.work_decomposition,
			smem_storage);

	}

	if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStop();
		kernel_stats.Flush();
	}
}

} // namespace expand_compact_atomic
} // namespace bfs
} // namespace graph
} // namespace b40c

