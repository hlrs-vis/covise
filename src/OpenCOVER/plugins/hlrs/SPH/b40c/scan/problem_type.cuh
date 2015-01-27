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
 * Scan problem type
 ******************************************************************************/

#pragma once

#include <b40c/reduction/problem_type.cuh>

namespace b40c {
namespace scan {


/**
 * Type of scan problem
 */
template <
	typename T,
	typename SizeT,
	typename ReductionOp,
	typename _IdentityOp,
	bool _EXCLUSIVE,				// Whether or not to perform an exclusive (vs. inclusive) prefix scan
	bool _COMMUTATIVE>				// Whether or not the associative scan operator is commutative vs. non-commuatative (the commutative-only implementation is generally faster)
struct ProblemType :
	reduction::ProblemType<T, SizeT, ReductionOp>	// Inherit from reduction problem type
{
	enum {
		EXCLUSIVE 			= _EXCLUSIVE,
		COMMUTATIVE 		= _COMMUTATIVE,
	};

	typedef _IdentityOp IdentityOp;					// Identity operator type
};


} // namespace scan
} // namespace b40c

