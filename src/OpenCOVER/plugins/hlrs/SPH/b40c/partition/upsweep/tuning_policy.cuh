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
 * Tuning policy for partitioning upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

namespace b40c {
namespace partition {
namespace upsweep {


/**
 * Partitioning upsweep reduction tuning policy .  This type encapsulates our
 * kernel-tuning parameters (they are reflected via the static fields).
 *
 * Note that the number of bins is considered a tuning parameter as opposed
 * to a problem-type parameter.  (We can simply enact subsequent sub-partitioning
 * passes to effect more bins).
 *  
 * The kernel is specialized for problem-type, SM-version, etc. by declaring
 * it with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in 
 * expanding/unrolling the kernel code for specific architectures and problem 
 * types.    
 */
template <
	// Problem type
	typename ProblemType,

	int _CUDA_ARCH,
	bool _CHECK_ALIGNMENT,
	int _LOG_BINS,
	int _LOG_SCHEDULE_GRANULARITY,
	int _MIN_CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	util::io::ld::CacheModifier _READ_MODIFIER,
	util::io::st::CacheModifier _WRITE_MODIFIER>

struct TuningPolicy : ProblemType
{
	enum {
		CUDA_ARCH									= _CUDA_ARCH,
		CHECK_ALIGNMENT								= _CHECK_ALIGNMENT,
		LOG_BINS									= _LOG_BINS,
		LOG_SCHEDULE_GRANULARITY					= _LOG_SCHEDULE_GRANULARITY,
		MIN_CTA_OCCUPANCY  							= _MIN_CTA_OCCUPANCY,
		LOG_THREADS 								= _LOG_THREADS,
		LOG_LOAD_VEC_SIZE  							= _LOG_LOAD_VEC_SIZE,
		LOG_LOADS_PER_TILE 							= _LOG_LOADS_PER_TILE,

		SCHEDULE_GRANULARITY						= 1 << LOG_SCHEDULE_GRANULARITY,
		THREADS										= 1 << LOG_THREADS,
		TILE_ELEMENTS								= 1 << (LOG_THREADS + LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE),
	};

	static const util::io::ld::CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const util::io::st::CacheModifier WRITE_MODIFIER 	= _WRITE_MODIFIER;
};


} // namespace upsweep
} // namespace partition
} // namespace b40c

