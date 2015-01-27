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
 * Radix sort downsweep scan kernel (scatter into bins)
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/device_intrinsics.cuh>

#include <b40c/radix_sort/downsweep/cta.cuh>

namespace b40c {
namespace radix_sort {
namespace downsweep {


/**
 * Radix sort downsweep scan pass (specialized for early-exit)
 */
template <typename KernelPolicy, bool EARLY_EXIT = KernelPolicy::EARLY_EXIT>
struct DownsweepPass
{
	static __device__ __forceinline__ void Invoke(
		int 								*&d_selectors,
		typename KernelPolicy::SizeT 		*&d_spine,
		typename KernelPolicy::KeyType 		*&d_keys0,
		typename KernelPolicy::KeyType 		*&d_keys1,
		typename KernelPolicy::ValueType 	*&d_values0,
		typename KernelPolicy::ValueType 	*&d_values1,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
		typename KernelPolicy::SmemStorage	&smem_storage)
	{
		typedef typename KernelPolicy::KeyType 				KeyType;
		typedef typename KernelPolicy::ValueType 			ValueType;
		typedef typename KernelPolicy::SizeT 				SizeT;
		typedef Cta<KernelPolicy> 							Cta;
		typedef typename KernelPolicy::Grid::LanePartial	LanePartial;

		LanePartial base_composite_counter = KernelPolicy::Grid::MyLanePartial(smem_storage.raking_lanes);
		int *raking_segment = 0;

		// Shared storage to help us choose which set of inputs to stream from
		__shared__ KeyType* 	d_keys[2];
		__shared__ ValueType* 	d_values[2];

		if (threadIdx.x < KernelPolicy::Grid::RAKING_THREADS) {

			// initalize lane warpscans
			int warpscan_lane = threadIdx.x >> KernelPolicy::Grid::LOG_RAKING_THREADS_PER_LANE;
			int warpscan_tid = threadIdx.x & (KernelPolicy::Grid::RAKING_THREADS_PER_LANE - 1);
			smem_storage.lanes_warpscan[warpscan_lane][0][warpscan_tid] = 0;

			raking_segment = KernelPolicy::Grid::MyRakingSegment(smem_storage.raking_lanes);

			// initialize bin warpscans
			if (threadIdx.x < KernelPolicy::BINS) {

				// Initialize bin_warpscan
				smem_storage.bin_warpscan[0][threadIdx.x] = 0;

				// We can early-exit if all keys go into the same bin (leave them as-is)
				const int SELECTOR_IDX 			= (KernelPolicy::CURRENT_PASS) & 0x1;
				const int NEXT_SELECTOR_IDX 	= (KernelPolicy::CURRENT_PASS + 1) & 0x1;

				// Determine where to read our input
				bool selector = (KernelPolicy::CURRENT_PASS == 0) ? 0 : d_selectors[SELECTOR_IDX];

				// Determine whether or not we have work to do and setup the next round
				// accordingly.  We can do this by looking at the first-block's
				// histograms and counting the number of bins with counts that are
				// non-zero and not-the-problem-size.
				if (KernelPolicy::PreprocessTraits::MustApply || KernelPolicy::PostprocessTraits::MustApply) {
					smem_storage.non_trivial_pass = true;
				} else {
					int first_block_carry = d_spine[util::FastMul(gridDim.x, threadIdx.x)];
					int predicate = ((first_block_carry > 0) && (first_block_carry < work_decomposition.num_elements));
					smem_storage.non_trivial_pass = util::WarpVoteAny<KernelPolicy::LOG_BINS>(predicate);
				}

				// Let the next round know which set of buffers to use
				if (blockIdx.x == 0) {
					d_selectors[NEXT_SELECTOR_IDX] = selector ^ smem_storage.non_trivial_pass;
				}

				// Determine our threadblock's work range
				work_decomposition.template GetCtaWorkLimits<
					KernelPolicy::LOG_TILE_ELEMENTS,
					KernelPolicy::LOG_SCHEDULE_GRANULARITY>(smem_storage.work_limits);

				d_keys[0] = (selector) ? d_keys1 : d_keys0;
				d_keys[1] = (selector) ? d_keys0 : d_keys1;
				d_values[0] = (selector) ? d_values1 : d_values0;
				d_values[1] = (selector) ? d_values0 : d_values1;
			}
		}

		// Sync to acquire non_trivial_pass, selector, and work limits
		__syncthreads();

		// Short-circuit this entire cycle
		if (!smem_storage.non_trivial_pass) return;

		Cta cta(
			smem_storage,
			d_keys[0],
			d_keys[1],
			d_values[0],
			d_values[1],
			d_spine,
			base_composite_counter,
			raking_segment);

		cta.ProcessWorkRange(smem_storage.work_limits);
	}
};


/**
 * Radix sort downsweep scan pass (specialized for non-early-exit)
 */
template <typename KernelPolicy>
struct DownsweepPass<KernelPolicy, false>
{
	static __device__ __forceinline__ void Invoke(
		int 								*&d_selectors,
		typename KernelPolicy::SizeT 		*&d_spine,
		typename KernelPolicy::KeyType 		*&d_keys0,
		typename KernelPolicy::KeyType 		*&d_keys1,
		typename KernelPolicy::ValueType 	*&d_values0,
		typename KernelPolicy::ValueType 	*&d_values1,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
		typename KernelPolicy::SmemStorage	&smem_storage)
	{
		typedef typename KernelPolicy::KeyType 				KeyType;
		typedef typename KernelPolicy::ValueType 			ValueType;
		typedef typename KernelPolicy::SizeT 				SizeT;
		typedef Cta<KernelPolicy> 							Cta;
		typedef typename KernelPolicy::Grid::LanePartial	LanePartial;

		LanePartial base_composite_counter = KernelPolicy::Grid::MyLanePartial(smem_storage.raking_lanes);
		int *raking_segment = 0;

		if (threadIdx.x < KernelPolicy::Grid::RAKING_THREADS) {

			// initalize lane warpscans
			int warpscan_lane = threadIdx.x >> KernelPolicy::Grid::LOG_RAKING_THREADS_PER_LANE;
			int warpscan_tid = threadIdx.x & (KernelPolicy::Grid::RAKING_THREADS_PER_LANE - 1);
			smem_storage.lanes_warpscan[warpscan_lane][0][warpscan_tid] = 0;

			raking_segment = KernelPolicy::Grid::MyRakingSegment(smem_storage.raking_lanes);

			// initialize bin warpscans
			if (threadIdx.x < KernelPolicy::BINS) {

				// Initialize bin_warpscan
				smem_storage.bin_warpscan[0][threadIdx.x] = 0;

				// Determine our threadblock's work range
				work_decomposition.template GetCtaWorkLimits<
					KernelPolicy::LOG_TILE_ELEMENTS,
					KernelPolicy::LOG_SCHEDULE_GRANULARITY>(smem_storage.work_limits);
			}
		}

		// Sync to acquire non_trivial_pass, selector, and work limits
		__syncthreads();

		if (KernelPolicy::CURRENT_PASS & 0x1) {

			// d_keys1 --> d_keys0
			Cta cta(
				smem_storage,
				d_keys1,
				d_keys0,
				d_values1,
				d_values0,
				d_spine,
				base_composite_counter,
				raking_segment);

			cta.ProcessWorkRange(smem_storage.work_limits);

		} else {

			// d_keys0 --> d_keys1
			Cta cta(
				smem_storage,
				d_keys0,
				d_keys1,
				d_values0,
				d_values1,
				d_spine,
				base_composite_counter,
				raking_segment);

			cta.ProcessWorkRange(smem_storage.work_limits);
		}
	}
};


/**
 * Radix sort downsweep scan kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__ 
void Kernel(
	int 								*d_selectors,
	typename KernelPolicy::SizeT 		*d_spine,
	typename KernelPolicy::KeyType 		*d_keys0,
	typename KernelPolicy::KeyType 		*d_keys1,
	typename KernelPolicy::ValueType 	*d_values0,
	typename KernelPolicy::ValueType 	*d_values1,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> work_decomposition)
{
	// Shared memory pool
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	DownsweepPass<KernelPolicy>::Invoke(
		d_selectors,
		d_spine,
		d_keys0,
		d_keys1,
		d_values0,
		d_values1,
		work_decomposition,
		smem_storage);
}



} // namespace downsweep
} // namespace radix_sort
} // namespace b40c

