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
 * Composite-counter functionality for partitioning upsweep reduction kernels
 ******************************************************************************/

#pragma once

namespace b40c {
namespace partition {
namespace upsweep {


/**
 * Shared-memory lanes of composite counters.
 *
 * We keep our per-thread composite counters in smem because we simply don't
 * have enough register storage.
 */
template <typename KernelPolicy>
struct CompostiteCounters
{
	enum {
		COMPOSITE_LANES = KernelPolicy::COMPOSITE_LANES,
	};


	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate lane
	 */
	template <int LANE, int dummy = 0>
	struct Iterate
	{
		// ResetCompositeCounters
		template <typename Cta>
		static __device__ __forceinline__ void ResetCompositeCounters(Cta *cta)
		{
			cta->smem_storage.composite_counters.words[LANE][threadIdx.x] = 0;
			Iterate<LANE + 1>::ResetCompositeCounters(cta);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<COMPOSITE_LANES, dummy>
	{
		// ResetCompositeCounters
		template <typename Cta>
		static __device__ __forceinline__ void ResetCompositeCounters(Cta *cta) {}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Resets our composite-counter lanes
	 */
	template <typename Cta>
	__device__ __forceinline__ void ResetCompositeCounters(Cta *cta)
	{
		Iterate<0>::ResetCompositeCounters(cta);
	}
};


} // namespace upsweep
} // namespace partition
} // namespace b40c

