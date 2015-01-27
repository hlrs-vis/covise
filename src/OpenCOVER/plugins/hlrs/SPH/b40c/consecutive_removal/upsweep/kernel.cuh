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
 * Consecutive removal upsweep reduction kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/consecutive_removal/upsweep/cta.cuh>

namespace b40c {
namespace consecutive_removal {
namespace upsweep {


/**
 * Consecutive removal upsweep reduction pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void UpsweepPass(
	typename KernelPolicy::KeyType								*d_in_keys,
	typename KernelPolicy::SizeT								*d_spine,
	typename KernelPolicy::EqualityOp							equality_op,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
	typename KernelPolicy::SmemStorage							&smem_storage)
{
	typedef Cta<KernelPolicy> 					Cta;
	typedef typename KernelPolicy::SizeT 		SizeT;

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		d_in_keys,
		d_spine,
		equality_op);

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
		KernelPolicy::LOG_TILE_ELEMENTS,
		KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// Quit if we're the last threadblock (no need for it in upsweep).
	if (work_limits.last_block) {
		return;
	}

	cta.ProcessWorkRange(work_limits);
}


/**
 * Consecutive reduction upsweep reduction kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::KeyType								*d_in_keys,
	typename KernelPolicy::SizeT								*d_spine,
	typename KernelPolicy::EqualityOp							equality_op,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	work_decomposition)
{
	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	UpsweepPass<KernelPolicy>(
		d_in_keys,
		d_spine,
		equality_op,
		work_decomposition,
		smem_storage);
}


} // namespace upsweep
} // namespace consecutive_removal
} // namespace b40c

