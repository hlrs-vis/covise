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
 * Unified radix sort policy
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/partition/policy.cuh>

#include <b40c/radix_sort/upsweep/tuning_policy.cuh>
#include <b40c/radix_sort/upsweep/kernel_policy.cuh>
#include <b40c/radix_sort/upsweep/kernel.cuh>
#include <b40c/radix_sort/downsweep/tuning_policy.cuh>
#include <b40c/radix_sort/downsweep/kernel_policy.cuh>
#include <b40c/radix_sort/downsweep/kernel.cuh>

#include <b40c/scan/problem_type.cuh>
#include <b40c/scan/kernel_policy.cuh>
#include <b40c/scan/spine/kernel.cuh>

namespace b40c {
namespace radix_sort {


/**
 * Unified radix sort policy type.
 *
 * In addition to kernel tuning parameters that guide the kernel compilation for
 * upsweep, spine, and downsweep kernels, this type includes enactor tuning
 * parameters that define kernel-dispatch policy.   By encapsulating all of the
 * kernel tuning policies, we assure operational consistency over an entire
 * partitioning pass.
 */
template <
	// Problem Type
	typename ProblemType,

	// Common
	int CUDA_ARCH,
	int _RADIX_BITS,
	int LOG_SCHEDULE_GRANULARITY,
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	bool EARLY_EXIT,
	bool _UNIFORM_SMEM_ALLOCATION,
	bool _UNIFORM_GRID_SIZE,
	bool _OVERSUBSCRIBED_GRID_SIZE,
	
	// Upsweep
	int UPSWEEP_MIN_CTA_OCCUPANCY,
	int UPSWEEP_LOG_THREADS,
	int UPSWEEP_LOG_LOAD_VEC_SIZE,
	int UPSWEEP_LOG_LOADS_PER_TILE,
	
	// Spine-scan
	int SPINE_LOG_THREADS,
	int SPINE_LOG_LOAD_VEC_SIZE,
	int SPINE_LOG_LOADS_PER_TILE,
	int SPINE_LOG_RAKING_THREADS,

	// Downsweep
	partition::downsweep::ScatterStrategy DOWNSWEEP_SCATTER_STRATEGY,
	int DOWNSWEEP_MIN_CTA_OCCUPANCY,
	int DOWNSWEEP_LOG_THREADS,
	int DOWNSWEEP_LOG_LOAD_VEC_SIZE,
	int DOWNSWEEP_LOG_LOADS_PER_CYCLE,
	int DOWNSWEEP_LOG_CYCLES_PER_TILE,
	int DOWNSWEEP_LOG_RAKING_THREADS>

struct Policy :
	partition::Policy<
		ProblemType,
		CUDA_ARCH,
		READ_MODIFIER,
		WRITE_MODIFIER,
		SPINE_LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		SPINE_LOG_RAKING_THREADS>
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename ProblemType::KeyType 		KeyType;
	typedef typename ProblemType::ValueType		ValueType;
	typedef typename ProblemType::SizeT 		SizeT;

	typedef void (*UpsweepKernelPtr)(int*, SizeT*, KeyType*, KeyType*, util::CtaWorkDistribution<SizeT>);
	typedef void (*DownsweepKernelPtr)(int*, SizeT*, KeyType*, KeyType*, ValueType*, ValueType*, util::CtaWorkDistribution<SizeT>);

	static const bool CHECK_ALIGNMENT = true;

	//---------------------------------------------------------------------
	// Tuning Policies
	//---------------------------------------------------------------------

	typedef upsweep::TuningPolicy<
		ProblemType,
		CUDA_ARCH,
		CHECK_ALIGNMENT,
		_RADIX_BITS,
		LOG_SCHEDULE_GRANULARITY,
		UPSWEEP_MIN_CTA_OCCUPANCY,
		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,
		UPSWEEP_LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		EARLY_EXIT>
			Upsweep;

	typedef downsweep::TuningPolicy<
		ProblemType,
		CUDA_ARCH,
		CHECK_ALIGNMENT,
		_RADIX_BITS,
		LOG_SCHEDULE_GRANULARITY,
		DOWNSWEEP_MIN_CTA_OCCUPANCY,
		DOWNSWEEP_LOG_THREADS,
		DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		DOWNSWEEP_LOG_LOADS_PER_CYCLE,
		DOWNSWEEP_LOG_CYCLES_PER_TILE,
		DOWNSWEEP_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		DOWNSWEEP_SCATTER_STRATEGY,
		EARLY_EXIT>
			Downsweep;

	//---------------------------------------------------------------------
	// Kernel function pointer retrieval
	//---------------------------------------------------------------------

	template <typename PassPolicy>
	static UpsweepKernelPtr UpsweepKernel() {
		return upsweep::Kernel<upsweep::KernelPolicy<typename Policy::Upsweep, PassPolicy> >;
	}

	template <typename PassPolicy>
	static DownsweepKernelPtr DownsweepKernel() {
		return downsweep::Kernel<downsweep::KernelPolicy<typename Policy::Downsweep, PassPolicy> >;
	}


	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------

	enum {
		RADIX_BITS					= _RADIX_BITS,
		UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
		OVERSUBSCRIBED_GRID_SIZE	= _OVERSUBSCRIBED_GRID_SIZE,
	};
};
		

}// namespace radix_sort
}// namespace b40c

