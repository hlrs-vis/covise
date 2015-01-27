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
 * Consecutive reduction spine scan kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/consecutive_reduction/spine/cta.cuh>

namespace b40c {
namespace consecutive_reduction {
namespace spine {


/**
 * Consecutive reduction spine scan pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void SpinePass(
	typename KernelPolicy::ValueType 		*d_in_partials,
	typename KernelPolicy::ValueType 		*d_out_partials,
	typename KernelPolicy::SizeT			*d_in_flags,
	typename KernelPolicy::SizeT			*d_out_flags,
	typename KernelPolicy::SpineSizeT 		spine_elements,
	typename KernelPolicy::ReductionOp 		reduction_op,
	typename KernelPolicy::SmemStorage		&smem_storage)
{
	typedef Cta<KernelPolicy> Cta;
	typedef typename KernelPolicy::SpineSizeT 			SpineSizeT;
	typedef typename KernelPolicy::SrtsSoaDetails 		SrtsSoaDetails;
	typedef typename KernelPolicy::SoaScanOperator		SoaScanOperator;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		d_in_partials,
		d_out_partials,
		d_in_flags,
		d_out_flags,
		SoaScanOperator(reduction_op));

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SpineSizeT guarded_elements = spine_elements & (KernelPolicy::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SpineSizeT guarded_offset = spine_elements - guarded_elements;

	util::CtaWorkLimits<SpineSizeT> work_limits(
		0,					// Offset at which this CTA begins processing
		spine_elements,		// Total number of elements for this CTA to process
		guarded_offset, 	// Offset of final, partially-full tile (requires guarded loads)
		guarded_elements,	// Number of elements in partially-full tile
		spine_elements,		// Offset at which this CTA is out-of-bounds
		true);				// If this block is the last block in the grid with any work

	cta.ProcessWorkRange(work_limits);
}


/**
 * Consecutive reduction spine scan kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__ 
void Kernel(
	typename KernelPolicy::ValueType 			*d_in_partials,
	typename KernelPolicy::ValueType 			*d_out_partials,
	typename KernelPolicy::SizeT				*d_in_flags,
	typename KernelPolicy::SizeT				*d_out_flags,
	typename KernelPolicy::SpineSizeT 			spine_elements,
	typename KernelPolicy::ReductionOp 			reduction_op)
{
	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	SpinePass<KernelPolicy>(
		d_in_partials,
		d_out_partials,
		d_in_flags,
		d_out_flags,
		spine_elements,
		reduction_op,
		smem_storage);
}


} // namespace spine
} // namespace consecutive_reduction
} // namespace b40c

