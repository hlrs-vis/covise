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
 * Spine reduction kernel
 ******************************************************************************/

#pragma once

#include <b40c/reduction/cta.cuh>

namespace b40c {
namespace reduction {
namespace spine {


/**
 * Spine reduction pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void SpinePass(
	typename KernelPolicy::T 			*d_in,
	typename KernelPolicy::T 			*d_spine,
	typename KernelPolicy::SizeT 		spine_elements,
	typename KernelPolicy::ReductionOp	reduction_op,
	typename KernelPolicy::SmemStorage	&smem_storage)
{
	typedef Cta<KernelPolicy> 				Cta;
	typedef typename KernelPolicy::T 		T;
	typedef typename KernelPolicy::SizeT 	SizeT;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		d_in,
		d_spine,
		reduction_op);

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SizeT guarded_elements = spine_elements & (KernelPolicy::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SizeT guarded_offset = spine_elements - guarded_elements;

	util::CtaWorkLimits<SizeT> work_limits(
		0,					// Offset at which this CTA begins processing
		spine_elements,		// Total number of elements for this CTA to process
		guarded_offset, 	// Offset of final, partially-full tile (requires guarded loads)
		guarded_elements,	// Number of elements in partially-full tile
		spine_elements,		// Offset at which this CTA is out-of-bounds
		true);				// If this block is the last block in the grid with any work

	cta.ProcessWorkRange(work_limits);
}


/**
 * Spine reduction kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__ 
void Kernel(
	typename KernelPolicy::T 				*d_in,
	typename KernelPolicy::T 				*d_spine,
	typename KernelPolicy::SizeT 			spine_elements,
	typename KernelPolicy::ReductionOp		reduction_op)

{
	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	SpinePass<KernelPolicy>(
		d_in,
		d_spine,
		spine_elements,
		reduction_op,
		smem_storage);
}

} // namespace spine
} // namespace reduction
} // namespace b40c

