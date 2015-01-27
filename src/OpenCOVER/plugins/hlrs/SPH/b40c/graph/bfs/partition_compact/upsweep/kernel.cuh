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
 * Upsweep kernel (bin reduction/counting)
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/graph/bfs/partition_compact/upsweep/cta.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace partition_compact {
namespace upsweep {


/**
 * Upsweep compaction pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void UpsweepPass(
	int												&num_gpus,
	typename KernelPolicy::VertexId 				*&d_in,
	typename KernelPolicy::ValidFlag				*&d_out_flag,
	typename KernelPolicy::SizeT 					*&d_spine,
	typename KernelPolicy::CollisionMask 			*&d_collision_cache,
	typename KernelPolicy::SmemStorage				&smem_storage)
{
	typedef Cta<KernelPolicy> 					Cta;
	typedef typename KernelPolicy::SizeT 		SizeT;

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	smem_storage.work_decomposition.template GetCtaWorkLimits<
		KernelPolicy::LOG_TILE_ELEMENTS,
		KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		num_gpus,
		d_in,
		d_out_flag,
		d_spine,
		d_collision_cache);

	// Process all tiles
	cta.ProcessWorkRange(work_limits);
}


/**
 * Upsweep compaction kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::VertexId			queue_index,
	int 									num_gpus,
	typename KernelPolicy::VertexId 		*d_in,
	typename KernelPolicy::ValidFlag		*d_out_flag,
	typename KernelPolicy::SizeT			*d_spine,
	typename KernelPolicy::CollisionMask 	*d_collision_cache,
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

	// Determine work decomposition
	if (threadIdx.x == 0) {
		// Obtain problem size
		SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);

		// Initialize work decomposition in smem
		smem_storage.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
			num_elements, gridDim.x);
	}

	// Barrier to protect work decomposition
	__syncthreads();

	UpsweepPass<KernelPolicy>(
		num_gpus,
		d_in,
		d_out_flag,
		d_spine,
		d_collision_cache,
		smem_storage);

	if (KernelPolicy::INSTRUMENT) {
		if (threadIdx.x == 0) {
			kernel_stats.MarkStop();
		}
	}
}


} // namespace upsweep
} // namespace partition_compact
} // namespace bfs
} // namespace graph
} // namespace b40c

