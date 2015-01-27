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
 * Partitioning problem type
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace partition {


/**
 * Type of partitioning problem (i.e., data types to partition)
 */
template <
	typename _KeyType,
	typename _ValueType,
	typename _SizeT>
struct ProblemType
{
	// The type of data we are operating upon
	typedef _KeyType 		KeyType;
	typedef _ValueType 		ValueType;

	// The integer type we should use to index into data arrays (e.g., size_t, uint32, uint64, etc)
	typedef _SizeT 			SizeT;

	enum {
		KEYS_ONLY = util::Equals<ValueType, util::NullType>::VALUE,
	};
};
		

}// namespace partition
}// namespace b40c

