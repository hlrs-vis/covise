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
 * Unified scan policy
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/reduction/kernel_policy.cuh>

#include <b40c/scan/kernel_policy.cuh>
#include <b40c/scan/upsweep/kernel.cuh>
#include <b40c/scan/spine/kernel.cuh>
#include <b40c/scan/downsweep/kernel.cuh>

namespace b40c {
namespace scan {


/**
 * Unified scan policy type.
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
	bool _UNIFORM_SMEM_ALLOCATION,
	bool _UNIFORM_GRID_SIZE,
	bool _OVERSUBSCRIBED_GRID_SIZE,
	int LOG_SCHEDULE_GRANULARITY,

	// Upsweep tunable params
	int UPSWEEP_MIN_CTA_OCCUPANCY,
	int UPSWEEP_LOG_THREADS,
	int UPSWEEP_LOG_LOAD_VEC_SIZE,
	int UPSWEEP_LOG_LOADS_PER_TILE,
	int UPSWEEP_LOG_RAKING_THREADS,

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

	typedef typename ProblemType::T T;
	typedef typename ProblemType::SizeT SizeT;
	typedef typename ProblemType::ReductionOp ReductionOp;
	typedef typename ProblemType::IdentityOp IdentityOp;

	typedef void (*UpsweepKernelPtr)(T*, T*, ReductionOp, IdentityOp, util::CtaWorkDistribution<SizeT>);
	typedef void (*SpineKernelPtr)(T*, T*, SizeT, ReductionOp, IdentityOp);
	typedef void (*DownsweepKernelPtr)(T*, T*, T*, ReductionOp, IdentityOp, util::CtaWorkDistribution<SizeT>);
	typedef void (*SingleKernelPtr)(T*, T*, SizeT, ReductionOp, IdentityOp);

	//---------------------------------------------------------------------
	// Kernel Policies
	//---------------------------------------------------------------------

	// Kernel config for the upsweep reduction kernel
	typedef KernelPolicy <
		ProblemType,
		CUDA_ARCH,
		true,								// Check alignment
		UPSWEEP_MIN_CTA_OCCUPANCY,
		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,
		UPSWEEP_LOG_LOADS_PER_TILE,
		UPSWEEP_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		LOG_SCHEDULE_GRANULARITY>
			Upsweep;

	// Problem type for spine scan (ensures exclusive scan)
	typedef scan::ProblemType<
		T,
		SizeT,
		ReductionOp,
		IdentityOp,
		true,								// Exclusive
		ProblemType::COMMUTATIVE> SpineProblemType;

	// Kernel config for the spine scan kernel
	typedef KernelPolicy <
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

	// Kernel config for the downsweep scan kernel
	typedef KernelPolicy <
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
		LOG_SCHEDULE_GRANULARITY>
			Downsweep;

	// Kernel config for a one-level pass using the spine scan kernel
	typedef KernelPolicy <
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
		SPINE_LOG_LOADS_PER_TILE + SPINE_LOG_LOAD_VEC_SIZE + SPINE_LOG_THREADS>
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
		return spine::Kernel<Single>;
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

	static void Print()
	{
		// ProblemType type parameters
		printf("%d, ", sizeof(T));
		printf("%d, ", sizeof(SizeT));
		printf("%d, ", CUDA_ARCH);

		// Common tunable params
		printf("%s, ", CacheModifierToString(_READ_MODIFIER));
		printf("%s, ", CacheModifierToString(_WRITE_MODIFIER));
		printf("%s ", (_UNIFORM_SMEM_ALLOCATION) ? "true" : "false");
		printf("%s ", (_UNIFORM_GRID_SIZE) ? "true" : "false");
		printf("%s ", (_OVERSUBSCRIBED_GRID_SIZE) ? "true" : "false");
		printf("%d, ", LOG_SCHEDULE_GRANULARITY);

		// Upsweep tunable params
		printf("%d, ", UPSWEEP_MIN_CTA_OCCUPANCY);
		printf("%d, ", UPSWEEP_LOG_THREADS);
		printf("%d, ", UPSWEEP_LOG_LOAD_VEC_SIZE);
		printf("%d, ", UPSWEEP_LOG_LOADS_PER_TILE);
		printf("%d, ", UPSWEEP_LOG_RAKING_THREADS);

		// Spine tunable params
		printf("%d, ", SPINE_LOG_THREADS);
		printf("%d, ", SPINE_LOG_LOAD_VEC_SIZE);
		printf("%d, ", SPINE_LOG_LOADS_PER_TILE);
		printf("%d, ", SPINE_LOG_RAKING_THREADS);

		// Upsweep tunable params
		printf("%d, ", DOWNSWEEP_MIN_CTA_OCCUPANCY);
		printf("%d, ", DOWNSWEEP_LOG_THREADS);
		printf("%d, ", DOWNSWEEP_LOG_LOAD_VEC_SIZE);
		printf("%d, ", DOWNSWEEP_LOG_LOADS_PER_TILE);
		printf("%d, ", DOWNSWEEP_LOG_RAKING_THREADS);
	}
};
		

}// namespace scan
}// namespace b40c

