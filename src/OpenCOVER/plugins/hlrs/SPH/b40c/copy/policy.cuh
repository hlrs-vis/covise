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
 * Copy configuration policy
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/copy/kernel.cuh>

namespace b40c {
namespace copy {


/**
 * A detailed policy type that specializes kernel and dispatch
 * code for a specific copy pass. It encapsulates our kernel-tuning
 * parameters (they are reflected via the static fields).
 *
 * The kernel is specialized for problem-type, SM-version, etc. by declaring
 * it with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 */
template <
	// ProblemType type parameters
	typename _T,
	typename _SizeT,

	// Machine parameters
	int CUDA_ARCH,

	// Tunable parameters
	int _LOG_SCHEDULE_GRANULARITY,
	int _MIN_CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	util::io::ld::CacheModifier _READ_MODIFIER,
	util::io::st::CacheModifier _WRITE_MODIFIER,
	bool _WORK_STEALING,
	bool _OVERSUBSCRIBED_GRID_SIZE>

struct Policy
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef _T 			T;
	typedef _SizeT 		SizeT;

	typedef void (*KernelPtr)(T*, T*, util::CtaWorkDistribution<SizeT>, util::CtaWorkProgress, int);

	//---------------------------------------------------------------------
	// Kernel function pointer retrieval
	//---------------------------------------------------------------------

	static KernelPtr Kernel() {
		return copy::Kernel<Policy>;
	}


	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------

	static const util::io::ld::CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const util::io::st::CacheModifier WRITE_MODIFIER 	= _WRITE_MODIFIER;

	enum {

		LOG_THREADS 					= _LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,

		LOG_LOAD_VEC_SIZE  				= _LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 				= _LOG_LOADS_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_LOAD_STRIDE					= LOG_THREADS + LOG_LOAD_VEC_SIZE,
		LOAD_STRIDE						= 1 << LOG_LOAD_STRIDE,

		LOG_WARPS						= LOG_THREADS - B40C_LOG_WARP_THREADS(CUDA_ARCH),
		WARPS							= 1 << LOG_WARPS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,

		LOG_SCHEDULE_GRANULARITY		= _LOG_SCHEDULE_GRANULARITY,
		SCHEDULE_GRANULARITY			= 1 << LOG_SCHEDULE_GRANULARITY,

		CHECK_ALIGNMENT					= 1,

		THREAD_OCCUPANCY				= B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		MAX_CTA_OCCUPANCY  				= B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), THREAD_OCCUPANCY),
		MIN_CTA_OCCUPANCY				= _MIN_CTA_OCCUPANCY,

		WORK_STEALING				 	= _WORK_STEALING,
		OVERSUBSCRIBED_GRID_SIZE		= _OVERSUBSCRIBED_GRID_SIZE,

		VALID 							= (MAX_CTA_OCCUPANCY > 0)
	};


	static void Print()
	{
		// ProblemType type parameters
		printf("%d, ", sizeof(T));
		printf("%d, ", sizeof(SizeT));
		printf("%d, ", CUDA_ARCH);

		// Tunable parameters
		printf("%d, ", _LOG_SCHEDULE_GRANULARITY);
		printf("%d, ", _MIN_CTA_OCCUPANCY);
		printf("%d, ", _LOG_THREADS);
		printf("%d, ", _LOG_LOAD_VEC_SIZE);
		printf("%d, ", _LOG_LOADS_PER_TILE);
		printf("%s, ", CacheModifierToString(_READ_MODIFIER));
		printf("%s, ", CacheModifierToString(_WRITE_MODIFIER));
		printf("%s, ", (WORK_STEALING) ? "true" : "false");
		printf("%s ", (_OVERSUBSCRIBED_GRID_SIZE) ? "true" : "false");
	}
};
		

}// namespace copy
}// namespace b40c

