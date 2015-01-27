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
 * Consecutive removal problem type
 ******************************************************************************/

#pragma once

namespace b40c {
namespace consecutive_removal {


/**
 * Type of consecutive removal problem
 */
template <
	typename _KeyType,
	typename _ValueType,
	typename _SizeT,
	typename _EqualityOp>
struct ProblemType
{
	// The type of data we are operating upon
	typedef _KeyType 		KeyType;
	typedef _ValueType 		ValueType;
	typedef _SizeT 			SizeT;
	typedef _EqualityOp		EqualityOp;

	// The size_t type of spine we're using
	typedef int 			SpineSizeT;

	enum {
		KEYS_ONLY = util::Equals<ValueType, util::NullType>::VALUE,
	};
};


} // namespace consecutive_removal
} // namespace b40c

