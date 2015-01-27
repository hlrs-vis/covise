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
 * Configuration policy for segmented scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/srts_grid.cuh>
#include <b40c/util/srts_soa_details.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/segmented_scan/soa_scan_operator.cuh>

namespace b40c {
namespace segmented_scan {

/**
 * A detailed kernel configuration policy type that specializes kernel
 * code for a specific segmented scan pass. It encapsulates our
 * kernel-tuning parameters (they are reflected via the static fields).
 *
 * The kernel is specialized for problem-type, SM-version, etc. by declaring
 * it with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 */
template <
	// ProblemType type parameters
	typename ProblemType,
	bool _FINAL_KERNEL,

	// Machine parameters
	int CUDA_ARCH,
	bool _CHECK_ALIGNMENT,

	// Tunable parameters
	int _MIN_CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	int _LOG_RAKING_THREADS,
	util::io::ld::CacheModifier _READ_MODIFIER,
	util::io::st::CacheModifier _WRITE_MODIFIER,
	int _LOG_SCHEDULE_GRANULARITY>

struct KernelPolicy : ProblemType
{
	typedef typename ProblemType::T 				T;
	typedef typename ProblemType::Flag 				Flag;
	typedef typename ProblemType::ReductionOp 		ReductionOp;
	typedef typename ProblemType::IdentityOp 		IdentityOp;

	typedef util::Tuple<T, Flag> 					TileTuple;			// Structure-of-array "slice" tuple for local SRTS scanning

	typedef SoaScanOperator<
		ReductionOp,
		IdentityOp,
		TileTuple> 									SoaScanOperator;	// Structure-of-array scan operator


	static const util::io::ld::CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const util::io::st::CacheModifier WRITE_MODIFIER 	= _WRITE_MODIFIER;

	enum {
		FINAL_KERNEL					= _FINAL_KERNEL,

		LOG_THREADS 					= _LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,

		LOG_LOAD_VEC_SIZE  				= _LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 				= _LOG_LOADS_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_LOAD_STRIDE					= LOG_THREADS + LOG_LOAD_VEC_SIZE,
		LOAD_STRIDE						= 1 << LOG_LOAD_STRIDE,

		LOG_RAKING_THREADS				= _LOG_RAKING_THREADS,
		RAKING_THREADS					= 1 << LOG_RAKING_THREADS,

		LOG_WARPS						= LOG_THREADS - B40C_LOG_WARP_THREADS(CUDA_ARCH),
		WARPS							= 1 << LOG_WARPS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,

		LOG_SCHEDULE_GRANULARITY		= _LOG_SCHEDULE_GRANULARITY,
		SCHEDULE_GRANULARITY			= 1 << LOG_SCHEDULE_GRANULARITY,

		CHECK_ALIGNMENT					= _CHECK_ALIGNMENT
	};

	//
	// We reduce the elements in registers, and then place that partial
	// scan into smem rows for further scan
	//

	// SRTS grid type for partials
	typedef util::SrtsGrid<
		CUDA_ARCH,
		T,										// Partial type
		LOG_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_RAKING_THREADS,						// Raking threads
		true>									// There are prefix dependences between lanes
			PartialsSrtsGrid;

	// SRTS grid type for flags
	typedef util::SrtsGrid<
		CUDA_ARCH,
		Flag,									// Partial type
		LOG_THREADS,							// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_RAKING_THREADS,						// Raking threads
		true>									// There are prefix dependences between lanes
			FlagsSrtsGrid;


	/**
	 * Shared memory structure
	 */
	struct SmemStorage
	{
		T 		partials_warpscan[2][B40C_WARP_THREADS(CUDA_ARCH)];
		T 		partials_raking_elements[PartialsSrtsGrid::TOTAL_RAKING_ELEMENTS];

		Flag 	flags_warpscan[2][B40C_WARP_THREADS(CUDA_ARCH)];
		Flag 	flags_raking_elements[FlagsSrtsGrid::TOTAL_RAKING_ELEMENTS];
	};


	enum {
		THREAD_OCCUPANCY				= B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		SMEM_OCCUPANCY					= B40C_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),

		MAX_CTA_OCCUPANCY  				= B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY)),
		MIN_CTA_OCCUPANCY 				= _MIN_CTA_OCCUPANCY,

		VALID 							= (MAX_CTA_OCCUPANCY > 0)
	};


	// Tuple type of SRTS grid types
	typedef util::Tuple<
		PartialsSrtsGrid,
		FlagsSrtsGrid> SrtsGridTuple;

	// Operational details type for SRTS grid type
	typedef util::SrtsSoaDetails<
		TileTuple,
		SrtsGridTuple> SrtsSoaDetails;
};


} // namespace segmented_scan
} // namespace b40c

