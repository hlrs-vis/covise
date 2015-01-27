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
 * Downsweep kernel (scatter into bins)
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/device_intrinsics.cuh>

#include <b40c/graph/bfs/partition_compact/downsweep/cta.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace partition_compact {
namespace downsweep {


/**
 * Downsweep BFS compaction pass
 */
template <typename KernelPolicy, typename SmemStorage>
__device__ __forceinline__ void DownsweepPass(
	typename KernelPolicy::VertexId 			&queue_index,
	int											&num_gpus,
	typename KernelPolicy::VertexId 			*&d_in,
	typename KernelPolicy::VertexId 			*&d_out,
	typename KernelPolicy::VertexId 			*&d_parent_in,
	typename KernelPolicy::VertexId 			*&d_parent_out,
	typename KernelPolicy::ValidFlag			*&d_flags_in,
	typename KernelPolicy::SizeT 				*&d_spine,
	util::CtaWorkProgress 						&work_progress,
	SmemStorage									&smem_storage,
	int											*raking_segment)
{
	typedef Cta<KernelPolicy> 							Cta;
	typedef typename KernelPolicy::SizeT 				SizeT;
	typedef typename KernelPolicy::Grid::LanePartial	LanePartial;

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	smem_storage.work_decomposition.template GetCtaWorkLimits<
		KernelPolicy::LOG_TILE_ELEMENTS,
		KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// Return if we have no work to do
	if (!work_limits.elements) {
		return;
	}

	// Location of base composite counter in SRTS grid
	LanePartial base_composite_counter =
		KernelPolicy::Grid::MyLanePartial(smem_storage.raking_lanes);

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		num_gpus,
		d_in,
		d_out,
		d_parent_in,
		d_parent_out,
		d_flags_in,
		d_spine,
		base_composite_counter,
		raking_segment);

	cta.ProcessWorkRange(work_limits);
}


/**
 * Downsweep scan-scatter kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::VertexId			queue_index,
	int										num_gpus,
	typename KernelPolicy::VertexId 		* d_in,
	typename KernelPolicy::VertexId 		* d_out,
	typename KernelPolicy::VertexId 		* d_parent_in,
	typename KernelPolicy::VertexId 		* d_parent_out,
	typename KernelPolicy::ValidFlag		* d_flags_in,
	typename KernelPolicy::SizeT			* d_spine,
	util::CtaWorkProgress 					work_progress,
	util::KernelRuntimeStats				kernel_stats = util::KernelRuntimeStats())
{
	typedef typename KernelPolicy::SizeT SizeT;

	// Shared storage for CTA processing
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	if (KernelPolicy::INSTRUMENT) {
		if (threadIdx.x == 0) {
			kernel_stats.MarkStart();
		}
	}

	// SRTS grid raking pointer
	int *raking_segment = NULL;

	if (threadIdx.x < KernelPolicy::Grid::RAKING_THREADS) {

		// Initalize lane warpscans
		int warpscan_lane = threadIdx.x >> KernelPolicy::Grid::LOG_RAKING_THREADS_PER_LANE;
		int warpscan_tid = threadIdx.x & (KernelPolicy::Grid::RAKING_THREADS_PER_LANE - 1);
		smem_storage.lanes_warpscan[warpscan_lane][0][warpscan_tid] = 0;

		raking_segment = KernelPolicy::Grid::MyRakingSegment(smem_storage.raking_lanes);

		// Initialize bin warpscans
		if (threadIdx.x < KernelPolicy::BINS) {

			// Initialize bin_warpscan
			smem_storage.bin_warpscan[0][threadIdx.x] = 0;

			// Determine our threadblock's work range
			smem_storage.work_decomposition.template GetCtaWorkLimits<
				KernelPolicy::LOG_TILE_ELEMENTS,
				KernelPolicy::LOG_SCHEDULE_GRANULARITY>(smem_storage.work_limits);

			// Determine work decomposition
			if (threadIdx.x == 0) {

				// Obtain problem size
				SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);

				// Initialize work decomposition in smem
				smem_storage.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
					num_elements, gridDim.x);
			}
		}
	}

	// Barrier to protect work decomposition
	__syncthreads();

	DownsweepPass<KernelPolicy>(
		queue_index,
		num_gpus,
		d_in,
		d_out,
		d_parent_in,
		d_parent_out,
		d_flags_in,
		d_spine,
		work_progress,
		smem_storage,
		raking_segment);

	if (KernelPolicy::INSTRUMENT) {
		if (threadIdx.x == 0) {
			kernel_stats.MarkStop();
		}
	}
}



} // namespace downsweep
} // namespace partition_compact
} // namespace bfs
} // namespace graph
} // namespace b40c
