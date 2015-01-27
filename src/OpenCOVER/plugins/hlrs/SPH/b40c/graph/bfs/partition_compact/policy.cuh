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
 * Unified BFS partition/compaction policy
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/partition/policy.cuh>
#include <b40c/partition/upsweep/tuning_policy.cuh>
#include <b40c/partition/downsweep/tuning_policy.cuh>

#include <b40c/graph/bfs/partition_compact/upsweep/kernel.cuh>
#include <b40c/graph/bfs/partition_compact/upsweep/kernel_policy.cuh>
#include <b40c/graph/bfs/partition_compact/downsweep/kernel.cuh>
#include <b40c/graph/bfs/partition_compact/downsweep/kernel_policy.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace partition_compact {

/**
 * Unified partition/compact policy type.
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

	// Machine parameters
	int CUDA_ARCH,

	// Behavioral control parameters
	bool INSTRUMENT,								// Whether or not we want instrumentation logic generated

	// Common tunable parameters
	int LOG_BINS,
	int LOG_SCHEDULE_GRANULARITY,
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,

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

	typedef typename ProblemType::VertexId 			VertexId;
	typedef typename ProblemType::ValidFlag			ValidFlag;
	typedef typename ProblemType::CollisionMask 	CollisionMask;
	typedef typename ProblemType::SizeT 			SizeT;

	typedef typename ProblemType::KeyType 			KeyType;
	typedef typename ProblemType::ValueType 		ValueType;

	//---------------------------------------------------------------------
	// Kernel Policies
	//---------------------------------------------------------------------

	typedef upsweep::KernelPolicy<
		partition::upsweep::TuningPolicy<
			ProblemType,
			CUDA_ARCH,
			false,								// Do not check alignment
			LOG_BINS,
			LOG_SCHEDULE_GRANULARITY,
			UPSWEEP_MIN_CTA_OCCUPANCY,
			UPSWEEP_LOG_THREADS,
			UPSWEEP_LOG_LOAD_VEC_SIZE,
			UPSWEEP_LOG_LOADS_PER_TILE,
			READ_MODIFIER,
			WRITE_MODIFIER>,
		INSTRUMENT>
			Upsweep;

	typedef downsweep::KernelPolicy<
		partition::downsweep::TuningPolicy<
			ProblemType,
			CUDA_ARCH,
			false,								// Do not check alignment
			LOG_BINS,
			LOG_SCHEDULE_GRANULARITY,
			DOWNSWEEP_MIN_CTA_OCCUPANCY,
			DOWNSWEEP_LOG_THREADS,
			DOWNSWEEP_LOG_LOAD_VEC_SIZE,
			DOWNSWEEP_LOG_LOADS_PER_CYCLE,
			DOWNSWEEP_LOG_CYCLES_PER_TILE,
			DOWNSWEEP_LOG_RAKING_THREADS,
			READ_MODIFIER,
			WRITE_MODIFIER,
			DOWNSWEEP_SCATTER_STRATEGY>,
		INSTRUMENT>
			Downsweep;

};


} // namespace partition_compact
} // namespace bfs
} // namespace graph
} // namespace b40c

