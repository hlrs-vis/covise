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
 * Scan operator for consecutive reduction problems
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace consecutive_reduction {

/**
 * Structure-of-array scan operator
 */
template <typename ReductionOp, typename TileTuple>
struct SoaScanOperator
{
	typedef typename TileTuple::T0 	ValueType;
	typedef typename TileTuple::T1 	FlagType;

	enum {
		IDENTITY_STRIDES = false,			// There is no "identity" region of warpscan storage exists for strides to index into
	};

	// ValueType reduction operator
	ReductionOp reduction_op;

	// Constructor
	__device__ __forceinline__ SoaScanOperator(ReductionOp reduction_op) :
		reduction_op(reduction_op)
	{}

	// SOA scan operator
	__device__ __forceinline__ TileTuple operator()(
		const TileTuple &first,
		const TileTuple &second)
	{
/*		NVBUGS XXX
		return TileTuple(
			(second.t1) ? second.t0 : reduction_op(first.t0, second.t0),
			first.t1 + second.t1);
*/
		if (second.t1) {
			return TileTuple(second.t0, first.t1 + second.t1);
		} else {
			return TileTuple(reduction_op(first.t0, second.t0), first.t1 + second.t1);
		}
	}

	// SOA identity operator
	__device__ __forceinline__ TileTuple operator()()
	{
		TileTuple retval;
		retval.t1 = 0;			// Flag identity
		return retval;
	}
};


} // namespace consecutive_reduction
} // namespace b40c

