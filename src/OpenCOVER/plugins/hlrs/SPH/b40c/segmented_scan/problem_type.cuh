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
 * Segmented scan problem type
 ******************************************************************************/

#pragma once

namespace b40c {
namespace segmented_scan {

/**
 * Type of segmented scan problem
 */
template <
	typename _T,			// Partial type
	typename _Flag,			// Flag type
	typename _SizeT,
	typename _ReductionOp,
	typename _IdentityOp,
	bool _EXCLUSIVE>
struct ProblemType
{
	typedef _T 				T;				// The type of data we are operating upon
	typedef _Flag 			Flag;			// The type of flag we're using
	typedef _SizeT 			SizeT;			// The integer type we should use to index into data arrays (e.g., size_t, uint32, uint64, etc)
	typedef _ReductionOp 	ReductionOp;	// The function or functor type for binary reduction (implements "T op(const T&, const T&)")
	typedef _IdentityOp 	IdentityOp;		// Identity operator type

	enum {
		EXCLUSIVE 			= _EXCLUSIVE,
	};
};


} // namespace segmented_scan
} // namespace b40c

