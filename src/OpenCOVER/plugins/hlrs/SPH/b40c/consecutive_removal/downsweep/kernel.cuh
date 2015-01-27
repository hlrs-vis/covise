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
 * Consecutive removal downsweep scan kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/srts_details.cuh>
#include <b40c/consecutive_removal/downsweep/cta.cuh>

namespace b40c {
namespace consecutive_removal {
namespace downsweep {


/**
 * Downsweep consecutive removal pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void DownsweepPass(
	typename KernelPolicy::KeyType 								*d_in_keys,
	typename KernelPolicy::KeyType								*d_out_keys,
	typename KernelPolicy::ValueType 							*d_in_values,
	typename KernelPolicy::ValueType 							*d_out_values,
	typename KernelPolicy::SizeT 								*d_spine,
	typename KernelPolicy::SizeT								*d_num_compacted,
	typename KernelPolicy::EqualityOp							equality_op,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
	typename KernelPolicy::SmemStorage							&smem_storage)
{
	typedef Cta<KernelPolicy> 						Cta;
	typedef typename KernelPolicy::SizeT 			SizeT;

	// We need the exclusive partial from our spine
	SizeT spine_partial = 0;
	if (d_spine != NULL) {
		util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(
			spine_partial, d_spine + blockIdx.x);
	}

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		d_in_keys,
		d_out_keys,
		d_in_values,
		d_out_values,
		d_num_compacted,
		equality_op,
		spine_partial);

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
		KernelPolicy::LOG_TILE_ELEMENTS,
		KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

	cta.ProcessWorkRange(work_limits);
}


/**
 * Downsweep consecutive removal kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::KeyType 								*d_in_keys,
	typename KernelPolicy::KeyType								*d_out_keys,
	typename KernelPolicy::ValueType 							*d_in_values,
	typename KernelPolicy::ValueType 							*d_out_values,
	typename KernelPolicy::SizeT 								*d_spine,
	typename KernelPolicy::SizeT								*d_num_compacted,
	typename KernelPolicy::EqualityOp							equality_op,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> work_decomposition)
{
	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	DownsweepPass<KernelPolicy>(
		d_in_keys,
		d_out_keys,
		d_in_values,
		d_out_values,
		d_spine,
		d_num_compacted,
		equality_op,
		work_decomposition,
		smem_storage);
}


} // namespace downsweep
} // namespace consecutive_removal
} // namespace b40c

