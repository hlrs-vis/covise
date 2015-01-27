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
 * Unified partitioning policy
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/partition/spine/kernel.cuh>

#include <b40c/scan/problem_type.cuh>
#include <b40c/scan/kernel_policy.cuh>

namespace b40c {
namespace partition {


/**
 * Unified partitioning policy type.
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
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	
	// Spine-scan
	int SPINE_LOG_THREADS,
	int SPINE_LOG_LOAD_VEC_SIZE,
	int SPINE_LOG_LOADS_PER_TILE,
	int SPINE_LOG_RAKING_THREADS>

struct Policy : ProblemType
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename ProblemType::SizeT 		SizeT;

	typedef void (*SpineKernelPtr)(SizeT*, SizeT*, int);

	//---------------------------------------------------------------------
	// Kernel Policies
	//---------------------------------------------------------------------

	// Problem type for spine scan
	typedef scan::ProblemType<
		SizeT,								// spine scan type T
		int,								// spine scan SizeT
		util::Sum<SizeT>,
		util::Sum<SizeT>,
		true,								// exclusive
		true> SpineProblemType;				// addition is commutative

	// Kernel config for spine scan
	typedef scan::KernelPolicy <
		SpineProblemType,
		CUDA_ARCH,
		false,								// do not check alignment
		1,									// only a single-CTA grid
		SPINE_LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		SPINE_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		SPINE_LOG_LOADS_PER_TILE + SPINE_LOG_LOAD_VEC_SIZE + SPINE_LOG_THREADS>
			Spine;

	//---------------------------------------------------------------------
	// Kernel function pointer retrieval
	//---------------------------------------------------------------------

	static SpineKernelPtr SpineKernel() {
		return partition::spine::Kernel<Spine>;
	}

};
		

}// namespace partition
}// namespace b40c

