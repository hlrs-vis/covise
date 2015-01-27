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
namespace segmented_scan {

/**
 * Structure-of-array scan operator
 */
template <
	typename ReductionOp,
	typename IdentityOp,
	typename TileTuple>
struct SoaScanOperator
{
	typedef typename TileTuple::T0 	T;
	typedef typename TileTuple::T1 	Flag;

	enum {
		IDENTITY_STRIDES = true,			// There is an "identity" region of warpscan storage exists for strides to index into
	};

	// Caller-supplied operators
	ReductionOp 		scan_op;
	IdentityOp 			identity_op;

	// Constructor
	__device__ __forceinline__ SoaScanOperator(
		ReductionOp scan_op,
		IdentityOp identity_op) :
			scan_op(scan_op),
			identity_op(identity_op)
	{}

	// SOA scan operator
	__device__ __forceinline__ TileTuple operator()(
		const TileTuple &first,
		const TileTuple &second)
	{
		if (second.t1) {
			return second;
		}

		return TileTuple(scan_op(first.t0, second.t0), first.t1);
	}

	// SOA identity operator
	__device__ __forceinline__ TileTuple operator()()
	{
		return TileTuple(
			identity_op(),				// Partial Identity
			0);							// Flag Identity
	}
};


} // namespace segmented_scan
} // namespace b40c

