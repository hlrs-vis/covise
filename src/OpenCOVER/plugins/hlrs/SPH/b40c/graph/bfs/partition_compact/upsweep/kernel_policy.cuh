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
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Upsweep kernel configuration policy
 ******************************************************************************/

#pragma once

#include <b40c/partition/upsweep/kernel_policy.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace partition_compact {
namespace upsweep {


/**
 * A detailed upsweep kernel configuration policy type that specializes kernel
 * code for a specific compaction pass. It encapsulates tuning configuration
 * policy details derived from TuningPolicy.
 */
template <
	typename TuningPolicy,				// Partition policy

	// Behavioral control parameters
	bool _INSTRUMENT>					// Whether or not we want instrumentation logic generated
struct KernelPolicy :
	partition::upsweep::KernelPolicy<TuningPolicy>
{
	typedef partition::upsweep::KernelPolicy<TuningPolicy> 		Base;			// Base class
	typedef typename TuningPolicy::VertexId 					VertexId;
	typedef typename TuningPolicy::SizeT 						SizeT;

	enum {
		WARPS = KernelPolicy::WARPS,
	};

	/**
	 * Shared storage
	 */
	struct SmemStorage : Base::SmemStorage
	{
		enum {
			WARP_HASH_ELEMENTS				= 128,
			CUDA_ARCH						= KernelPolicy::CUDA_ARCH,
		};

		// Shared work-processing limits
		util::CtaWorkDistribution<SizeT>	work_decomposition;
		VertexId 							vid_hashtable[WARPS][WARP_HASH_ELEMENTS];

		enum {
			// Amount of storage we can use for hashing scratch space under target occupancy
			FULL_OCCUPANCY_BYTES			= (B40C_SMEM_BYTES(CUDA_ARCH) / KernelPolicy::MAX_CTA_OCCUPANCY)
												- sizeof(typename Base::SmemStorage)
												- sizeof(util::CtaWorkDistribution<SizeT>)
												- sizeof(VertexId[WARPS][WARP_HASH_ELEMENTS])
												- 128,

			HISTORY_HASH_ELEMENTS			= FULL_OCCUPANCY_BYTES /sizeof(VertexId),
		};

		// General pool for hashing
		VertexId 							history[HISTORY_HASH_ELEMENTS];
	};



	enum {
		INSTRUMENT								= _INSTRUMENT,

		CUDA_ARCH 								= KernelPolicy::CUDA_ARCH,
		LOG_THREADS 							= KernelPolicy::LOG_THREADS,
		THREAD_OCCUPANCY						= B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		SMEM_OCCUPANCY							= B40C_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),

		MAX_CTA_OCCUPANCY  						= B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY)),

		VALID									= (MAX_CTA_OCCUPANCY > 0),
	};
};
	


} // namespace upsweep
} // namespace partition_compact
} // namespace bfs
} // namespace graph
} // namespace b40c


