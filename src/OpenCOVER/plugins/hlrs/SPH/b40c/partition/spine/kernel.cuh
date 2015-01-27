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
 * Partition spine scan kernel
 *
 * Requires a b40c::scan::KernelPolicy.
 ******************************************************************************/

#pragma once

#include <b40c/scan/spine/kernel.cuh>

namespace b40c {
namespace partition {
namespace spine {


/**
 * Consecutive removal spine scan kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__ 
void Kernel(
	typename KernelPolicy::T			*d_in,
	typename KernelPolicy::T			*d_out,
	typename KernelPolicy::SizeT 		spine_elements)
{
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	typename KernelPolicy::ReductionOp reduction_op;
	typename KernelPolicy::IdentityOp identity_op;

	scan::spine::SpinePass<KernelPolicy>(
		d_in,
		d_out,
		spine_elements,
		reduction_op,
		identity_op,
		smem_storage);
}

} // namespace spine
} // namespace partition
} // namespace b40c

