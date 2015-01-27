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
 * Reduction problem type
 ******************************************************************************/

#pragma once

namespace b40c {
namespace reduction {


/**
 * Type of reduction problem
 */
template <
	typename _T,
	typename _SizeT,
	typename _ReductionOp>
struct ProblemType
{
	// The type of data we are operating upon
	typedef _T T;

	// The integer type we should use to index into data arrays (e.g., size_t, uint32, uint64, etc)
	typedef _SizeT SizeT;

	// The function or functor type for binary reduction (implements "T op(const T&, const T&)")
	typedef _ReductionOp ReductionOp;
};
		

}// namespace reduction
}// namespace b40c

