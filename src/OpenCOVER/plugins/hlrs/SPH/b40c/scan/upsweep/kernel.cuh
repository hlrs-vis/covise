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
 * Upsweep kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>

#include <b40c/reduction/cta.cuh>
#include <b40c/scan/upsweep/cta.cuh>

namespace b40c {
namespace scan {
namespace upsweep {


/**
 * Upsweep reduction pass (specialized to support non-commutative operators)
 */
template <
	typename KernelPolicy,
	bool COMMUTATIVE = KernelPolicy::COMMUTATIVE>
struct UpsweepPass
{
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::T 									*d_in,
		typename KernelPolicy::T 									*d_out,
		typename KernelPolicy::ReductionOp 							scan_op,
		typename KernelPolicy::IdentityOp 							identity_op,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
		typename KernelPolicy::SmemStorage							&smem_storage)
	{
		typedef Cta<KernelPolicy>					Cta;
		typedef typename KernelPolicy::SizeT 		SizeT;

		// CTA processing abstraction
		Cta cta(
			smem_storage,
			d_in,
			d_out,
			scan_op,
			identity_op);

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
};


/**
 * Upsweep reduction pass (specialized for commutative operators)
 */
template <typename KernelPolicy>
struct UpsweepPass<KernelPolicy, true>
{
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::T 									*d_in,
		typename KernelPolicy::T 									*d_out,
		typename KernelPolicy::ReductionOp 							scan_op,
		typename KernelPolicy::IdentityOp 							identity_op,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
		typename KernelPolicy::SmemStorage							&smem_storage)
	{
		typedef reduction::Cta<KernelPolicy>		Cta;
		typedef typename KernelPolicy::SizeT 		SizeT;

		// CTA processing abstraction
		Cta cta(
			smem_storage,
			d_in,
			d_out,
			scan_op);

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
};



/******************************************************************************
 * Upsweep Reduction Kernel Entrypoint
 ******************************************************************************/

/**
 * Upsweep reduction kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::T 									*d_in,
	typename KernelPolicy::T 									*d_spine,
	typename KernelPolicy::ReductionOp 							scan_op,
	typename KernelPolicy::IdentityOp 							identity_op,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	work_decomposition)
{
	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	UpsweepPass<KernelPolicy>::Invoke(
		d_in,
		d_spine,
		scan_op,
		identity_op,
		work_decomposition,
		smem_storage);
}


} // namespace upsweep
} // namespace scan
} // namespace b40c

