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
 * Upsweep reduction kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>

#include <b40c/reduction/cta.cuh>

namespace b40c {
namespace reduction {
namespace upsweep {


/**
 * Atomically steals work from a global work progress construct
 */
template <typename SizeT>
__device__ __forceinline__ SizeT StealWork(
	util::CtaWorkProgress &work_progress,
	int count)
{
	__shared__ SizeT s_offset;		// The offset at which this CTA performs tile processing, shared by all

	// Thread zero atomically steals work from the progress counter
	if (threadIdx.x == 0) {
		s_offset = work_progress.Steal<SizeT>(count);
	}

	__syncthreads();		// Protect offset

	return s_offset;
}


/**
 * Upsweep reduction pass (non-workstealing specialization)
 */
template <typename KernelPolicy, bool WORK_STEALING = KernelPolicy::WORK_STEALING>
struct UpsweepPass
{
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::T 									*d_in,
		typename KernelPolicy::T 									*d_out,
		typename KernelPolicy::ReductionOp							reduction_op,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
		util::CtaWorkProgress 										&work_progress,
		typename KernelPolicy::SmemStorage							&smem_storage)
	{
		typedef Cta<KernelPolicy> 				Cta;
		typedef typename KernelPolicy::SizeT 	SizeT;

		// CTA processing abstraction
		Cta cta(
			smem_storage,
			d_in,
			d_out,
			reduction_op);

		// Determine our threadblock's work range
		util::CtaWorkLimits<SizeT> work_limits;
		work_decomposition.template GetCtaWorkLimits<
			KernelPolicy::LOG_TILE_ELEMENTS,
			KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

		cta.ProcessWorkRange(work_limits);
	}
};


/**
 * Upsweep reduction pass (workstealing specialization)
 */
template <typename KernelPolicy>
struct UpsweepPass <KernelPolicy, true>
{
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::T 									*d_in,
		typename KernelPolicy::T 									*d_out,
		typename KernelPolicy::ReductionOp							reduction_op,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
		util::CtaWorkProgress 										&work_progress,
		typename KernelPolicy::SmemStorage							&smem_storage)
	{
		typedef Cta<KernelPolicy> 				Cta;
		typedef typename KernelPolicy::SizeT 	SizeT;

		// CTA processing abstraction
		Cta cta(
			smem_storage,
			d_in,
			d_out,
			reduction_op);

		// First CTA resets the work progress for the next pass
		if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
			work_progress.template PrepResetSteal<SizeT>();
		}

		// Total number of elements in full tiles
		SizeT unguarded_elements = work_decomposition.num_elements & (~(KernelPolicy::TILE_ELEMENTS - 1));

		// Each CTA needs to process at least one partial block of
		// input (otherwise our spine scan will be invalid)

		SizeT offset = blockIdx.x << KernelPolicy::LOG_TILE_ELEMENTS;
		if (offset < unguarded_elements) {

			// Process our one full tile (first tile seen)
			cta.template ProcessFullTile<true>(offset);

			// Determine the swath we just did
			SizeT swath = work_decomposition.grid_size << KernelPolicy::LOG_TILE_ELEMENTS;

			// Worksteal subsequent full tiles, if any
			while ((offset = StealWork<SizeT>(
				work_progress,
				KernelPolicy::TILE_ELEMENTS) + swath) < unguarded_elements)
			{
				cta.template ProcessFullTile<false>(offset);
			}

			// If the problem is big enough for the last CTA to be in this if-then-block,
			// have it do the remaining guarded work (not first tile)
			if (blockIdx.x == gridDim.x - 1) {
				cta.template ProcessPartialTile<false>(unguarded_elements, work_decomposition.num_elements);
			}

			// Collectively reduce accumulated carry from each thread into output
			// destination (all thread have valid reduction partials)
			cta.OutputToSpine();

		} else {

			// Last CTA does any extra, guarded work (first tile seen)
			cta.template ProcessPartialTile<true>(unguarded_elements, work_decomposition.num_elements);

			// Collectively reduce accumulated carry from each thread into output
			// destination (not every thread may have a valid reduction partial)
			cta.OutputToSpine(work_decomposition.num_elements - unguarded_elements);
		}
	}
};


/**
 * Upsweep reduction kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::T 									*d_in,
	typename KernelPolicy::T 									*d_spine,
	typename KernelPolicy::ReductionOp							reduction_op,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	work_decomposition,
	util::CtaWorkProgress										work_progress)
{
	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	UpsweepPass<KernelPolicy>::Invoke(
		d_in,
		d_spine,
		reduction_op,
		work_decomposition,
		work_progress,
		smem_storage);
}


} // namespace upsweep
} // namespace reduction
} // namespace b40c

