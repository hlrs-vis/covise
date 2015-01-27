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
 * Radix sort upsweep reduction kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/radix_sort/upsweep/cta.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {


/**
 * Radix sort upsweep reduction pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void UpsweepPass(
	int 									*&d_selectors,
	typename KernelPolicy::SizeT 			*&d_spine,
	typename KernelPolicy::KeyType 			*&d_in_keys,
	typename KernelPolicy::KeyType 			*&d_out_keys,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
	typename KernelPolicy::SmemStorage		&smem_storage)
{
	typedef Cta<KernelPolicy> 						Cta;
	typedef typename KernelPolicy::KeyType 			KeyType;
	typedef typename KernelPolicy::SizeT 			SizeT;
	
	// Determine where to read our input

	bool selector = ((KernelPolicy::EARLY_EXIT) && ((KernelPolicy::CURRENT_PASS != 0) && (d_selectors[KernelPolicy::CURRENT_PASS & 0x1]))) ||
		(KernelPolicy::CURRENT_PASS & 0x1);
	KeyType *d_keys = (selector) ? d_out_keys : d_in_keys;

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
		KernelPolicy::LOG_TILE_ELEMENTS,
		KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		d_keys,
		d_spine);
	
	// Accumulate digit counts for all tiles
	cta.ProcessWorkRange(work_limits);
}


/**
 * Radix sort upsweep reduction kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	int 								*d_selectors,
	typename KernelPolicy::SizeT 		*d_spine,
	typename KernelPolicy::KeyType 		*d_in_keys,
	typename KernelPolicy::KeyType 		*d_out_keys,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> work_decomposition)
{
	// Shared memory pool
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	UpsweepPass<KernelPolicy>(
		d_selectors,
		d_spine,
		d_in_keys,
		d_out_keys,
		work_decomposition,
		smem_storage);
}


} // namespace upsweep
} // namespace radix_sort
} // namespace b40c

