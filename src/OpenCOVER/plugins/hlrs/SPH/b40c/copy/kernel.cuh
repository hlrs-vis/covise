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
 * Copy kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/copy/cta.cuh>

namespace b40c {
namespace copy {


/**
 * Copy pass (non-workstealing)
 */
template <typename KernelPolicy, bool WORK_STEALING>
struct SweepPass
{
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::T 									*&d_in,
		typename KernelPolicy::T 									*&d_out,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
		util::CtaWorkProgress 										&work_progress,
		int 														&extra_bytes)
	{
		typedef Cta<KernelPolicy> 				Cta;
		typedef typename KernelPolicy::T 		T;
		typedef typename KernelPolicy::SizeT 	SizeT;

		// CTA processing abstraction
		Cta cta(d_in, d_out);

		// Determine our threadblock's work range
		util::CtaWorkLimits<SizeT> work_limits;
		work_decomposition.template GetCtaWorkLimits<
			KernelPolicy::LOG_TILE_ELEMENTS,
			KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

		// Process full tiles of tile_elements
		while (work_limits.offset < work_limits.guarded_offset) {

			cta.ProcessTile(work_limits.offset);
			work_limits.offset += KernelPolicy::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-io
		if (work_limits.guarded_elements) {
			cta.ProcessTile(
				work_limits.offset,
				work_limits.guarded_elements);
		}

		// Cleanup any extra bytes
		if ((sizeof(typename KernelPolicy::T) > 1) && (blockIdx.x == gridDim.x - 1) && (threadIdx.x < extra_bytes)) {

			unsigned char* d_in_bytes = (unsigned char *)(d_in + work_limits.guarded_elements);
			unsigned char* d_out_bytes = (unsigned char *)(d_out + work_limits.guarded_elements);
			unsigned char extra_byte;

			util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(
				extra_byte, d_in_bytes + threadIdx.x);
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				extra_byte, d_out_bytes + threadIdx.x);
		}

	}
};


/**
 * Copy pass (workstealing)
 */
template <typename KernelPolicy>
struct SweepPass <KernelPolicy, true>
{
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::T 									*&d_in,
		typename KernelPolicy::T 									*&d_out,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
		util::CtaWorkProgress 										&work_progress,
		int 														&extra_bytes)
	{
		typedef Cta<KernelPolicy> 				Cta;
		typedef typename KernelPolicy::T 		T;
		typedef typename KernelPolicy::SizeT 	SizeT;

		// CTA processing abstraction
		Cta cta(d_in, d_out);

		// The offset at which this CTA performs tile processing
		__shared__ SizeT offset;

		// First CTA resets the work progress for the next pass
		if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
			work_progress.template PrepResetSteal<SizeT>();
		}

		// Steal full-tiles of work, incrementing progress counter
		SizeT unguarded_elements = work_decomposition.num_elements & (~(KernelPolicy::TILE_ELEMENTS - 1));
		while (true) {

			// Thread zero atomically steals work from the progress counter
			if (threadIdx.x == 0) {
				offset = work_progress.template Steal<SizeT>(KernelPolicy::TILE_ELEMENTS);
			}

			__syncthreads();		// Protect offset

			if (offset >= unguarded_elements) {
				// All done
				break;
			}

			cta.ProcessTile(offset);
		}

		// Last CTA does any extra, guarded work
		if (blockIdx.x == gridDim.x - 1) {

			SizeT guarded_elements = work_decomposition.num_elements - unguarded_elements;
			cta.ProcessTile(unguarded_elements, guarded_elements);

			// Cleanup any extra bytes
			if ((sizeof(typename KernelPolicy::T) > 1) && (threadIdx.x < extra_bytes)) {

				unsigned char* d_in_bytes = (unsigned char *)(d_in + work_decomposition.num_elements);
				unsigned char* d_out_bytes = (unsigned char *)(d_out + work_decomposition.num_elements);
				unsigned char extra_byte;

				util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(extra_byte, d_in_bytes + threadIdx.x);
				util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(extra_byte, d_out_bytes + threadIdx.x);
			}
		}
	}
};


/**
 *  Copy kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::T 									*d_in,
	typename KernelPolicy::T 									*d_out,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	work_decomposition,
	util::CtaWorkProgress 										work_progress,
	int 														extra_bytes)
{
	SweepPass<KernelPolicy, KernelPolicy::WORK_STEALING>::Invoke(
		d_in,
		d_out,
		work_decomposition,
		work_progress,
		extra_bytes);
}


} // namespace copy
} // namespace b40c

