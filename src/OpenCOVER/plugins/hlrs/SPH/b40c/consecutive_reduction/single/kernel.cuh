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
 * Consecutive removal single-CTA scan kernel
 ******************************************************************************/

#pragma once

#include <b40c/consecutive_reduction/downsweep/cta.cuh>

namespace b40c {
namespace consecutive_reduction {
namespace single {


/**
 *  Consecutive removal single-CTA scan pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void SinglePass(
	typename KernelPolicy::KeyType			*d_in_keys,
	typename KernelPolicy::KeyType			*d_out_keys,
	typename KernelPolicy::ValueType		*d_in_values,
	typename KernelPolicy::ValueType		*d_out_values,
	typename KernelPolicy::SizeT			*d_num_compacted,
	typename KernelPolicy::SizeT 			num_elements,
	typename KernelPolicy::ReductionOp 		reduction_op,
	typename KernelPolicy::EqualityOp		equality_op,
	typename KernelPolicy::SmemStorage		&smem_storage)
{
	typedef downsweep::Cta<KernelPolicy> 			Cta;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::SoaScanOperator	SoaScanOperator;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		d_in_keys,
		d_out_keys,
		d_in_values,
		d_out_values,
		d_num_compacted,
		SoaScanOperator(reduction_op),
		equality_op);

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SizeT guarded_elements = num_elements & (KernelPolicy::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SizeT guarded_offset = num_elements - guarded_elements;

	util::CtaWorkLimits<SizeT> work_limits(
		0,					// Offset at which this CTA begins processing
		num_elements,		// Total number of elements for this CTA to process
		guarded_offset, 	// Offset of final, partially-full tile (requires guarded loads)
		guarded_elements,	// Number of elements in partially-full tile
		num_elements,		// Offset at which this CTA is out-of-bounds
		true);				// If this block is the last block in the grid with any work

	cta.ProcessWorkRange(work_limits);
}


/**
 * Consecutive removal single-CTA scan kernel entrypoint
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__ 
void Kernel(
	typename KernelPolicy::KeyType			*d_in_keys,
	typename KernelPolicy::KeyType			*d_out_keys,
	typename KernelPolicy::ValueType		*d_in_values,
	typename KernelPolicy::ValueType		*d_out_values,
	typename KernelPolicy::SizeT			*d_num_compacted,
	typename KernelPolicy::SizeT 			num_elements,
	typename KernelPolicy::ReductionOp 		reduction_op,
	typename KernelPolicy::EqualityOp		equality_op)
{
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	SinglePass<KernelPolicy>(
		d_in_keys,
		d_out_keys,
		d_in_values,
		d_out_values,
		d_num_compacted,
		num_elements,
		reduction_op,
		equality_op,
		smem_storage);
}

} // namespace single
} // namespace consecutive_reduction
} // namespace b40c

