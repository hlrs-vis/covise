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
 * Unified consecutive removal policy
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/operators.cuh>

#include <b40c/consecutive_removal/upsweep/kernel.cuh>
#include <b40c/consecutive_removal/upsweep/kernel_policy.cuh>
#include <b40c/consecutive_removal/downsweep/kernel.cuh>
#include <b40c/consecutive_removal/downsweep/kernel_policy.cuh>
#include <b40c/consecutive_removal/spine/kernel.cuh>
#include <b40c/consecutive_removal/single/kernel.cuh>

#include <b40c/scan/kernel_policy.cuh>
#include <b40c/scan/problem_type.cuh>

namespace b40c {
namespace consecutive_removal {


/**
 * Unified consecutive removal policy type.
 *
 * In addition to kernel tuning parameters that guide the kernel compilation for
 * upsweep, spine, and downsweep kernels, this type includes enactor tuning
 * parameters that define kernel-dispatch policy.   By encapsulating all of the
 * kernel tuning policies, we assure operational consistency across all kernels.
 */
template <
	// ProblemType type parameters
	typename ProblemType,

	// Machine parameters
	int CUDA_ARCH,

	// Common tunable params
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	bool TWO_PHASE_SCATTER,
	bool CONSECUTIVE_SMEM_ASSIST,
	bool _UNIFORM_SMEM_ALLOCATION,
	bool _UNIFORM_GRID_SIZE,
	bool _OVERSUBSCRIBED_GRID_SIZE,
	int LOG_SCHEDULE_GRANULARITY,

	// Upsweep tunable params
	int UPSWEEP_MIN_CTA_OCCUPANCY,
	int UPSWEEP_LOG_THREADS,
	int UPSWEEP_LOG_LOAD_VEC_SIZE,
	int UPSWEEP_LOG_LOADS_PER_TILE,

	// Spine tunable params
	int SPINE_LOG_THREADS,
	int SPINE_LOG_LOAD_VEC_SIZE,
	int SPINE_LOG_LOADS_PER_TILE,
	int SPINE_LOG_RAKING_THREADS,

	// Downsweep tunable params
	int DOWNSWEEP_MIN_CTA_OCCUPANCY,
	int DOWNSWEEP_LOG_THREADS,
	int DOWNSWEEP_LOG_LOAD_VEC_SIZE,
	int DOWNSWEEP_LOG_LOADS_PER_TILE,
	int DOWNSWEEP_LOG_RAKING_THREADS>

struct Policy : ProblemType
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename ProblemType::KeyType 			KeyType;
	typedef typename ProblemType::ValueType			ValueType;
	typedef typename ProblemType::SizeT 			SizeT;
	typedef typename ProblemType::EqualityOp		EqualityOp;
	typedef typename ProblemType::SpineSizeT		SpineSizeT;

	typedef void (*UpsweepKernelPtr)(KeyType*, SizeT*, EqualityOp, util::CtaWorkDistribution<SizeT>);
	typedef void (*SpineKernelPtr)(SizeT*, SizeT*, SpineSizeT);
	typedef void (*DownsweepKernelPtr)(KeyType*, KeyType*, ValueType*, ValueType*, SizeT*, SizeT*, EqualityOp, util::CtaWorkDistribution<SizeT>);
	typedef void (*SingleKernelPtr)(KeyType*, KeyType*, ValueType*, ValueType*, SizeT*, SizeT, EqualityOp);

	// Kernel config for the upsweep reduction kernel
	typedef upsweep::KernelPolicy<
		ProblemType,
		CUDA_ARCH,
		true,								// Check alignment
		UPSWEEP_MIN_CTA_OCCUPANCY,
		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,
		UPSWEEP_LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		LOG_SCHEDULE_GRANULARITY,
		CONSECUTIVE_SMEM_ASSIST>
			Upsweep;

	// Problem type for spine
	typedef scan::ProblemType<
		SizeT,
		SpineSizeT,
		util::Sum<SizeT>,
		util::Sum<SizeT>,
		true,								// Exclusive
		true> SpineProblemType;				// Addition is commutative

	// Kernel config for the spine consecutive removal kernel
	typedef scan::KernelPolicy <
		SpineProblemType,
		CUDA_ARCH,
		false,								// Do not check alignment
		1,									// Only a single-CTA grid
		SPINE_LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		SPINE_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		SPINE_LOG_LOADS_PER_TILE + SPINE_LOG_LOAD_VEC_SIZE + SPINE_LOG_THREADS>
			Spine;

	// Kernel config for downsweep
	typedef downsweep::KernelPolicy <
		ProblemType,
		CUDA_ARCH,
		true,								// Check alignment
		DOWNSWEEP_MIN_CTA_OCCUPANCY,
		DOWNSWEEP_LOG_THREADS,
		DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		DOWNSWEEP_LOG_LOADS_PER_TILE,
		DOWNSWEEP_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		LOG_SCHEDULE_GRANULARITY,
		TWO_PHASE_SCATTER,
		CONSECUTIVE_SMEM_ASSIST>
			Downsweep;

	// Kernel config for single-cta (single-grid) consecutive duplicate removal
	typedef downsweep::KernelPolicy <
		ProblemType,
		CUDA_ARCH,
		true,								// Check alignment
		1,									// Only a single-CTA grid
		SPINE_LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		SPINE_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		SPINE_LOG_LOADS_PER_TILE + SPINE_LOG_LOAD_VEC_SIZE + SPINE_LOG_THREADS,
		TWO_PHASE_SCATTER,
		CONSECUTIVE_SMEM_ASSIST>
			Single;

	//---------------------------------------------------------------------
	// Kernel function pointer retrieval
	//---------------------------------------------------------------------

	static UpsweepKernelPtr UpsweepKernel() {
		return upsweep::Kernel<Upsweep>;
	}

	static SpineKernelPtr SpineKernel() {
		return spine::Kernel<Spine>;
	}

	static DownsweepKernelPtr DownsweepKernel() {
		return downsweep::Kernel<Downsweep>;
	}

	static SingleKernelPtr SingleKernel() {
		return single::Kernel<Single>;
	}


	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------

	enum {
		UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
		OVERSUBSCRIBED_GRID_SIZE	= _OVERSUBSCRIBED_GRID_SIZE,
		VALID 						= Upsweep::VALID && Spine::VALID && Downsweep::VALID && Single::VALID &&
										(Upsweep::LOG_TILE_ELEMENTS <= LOG_SCHEDULE_GRANULARITY),
	};

};
		

}// namespace consecutive_removal
}// namespace b40c

