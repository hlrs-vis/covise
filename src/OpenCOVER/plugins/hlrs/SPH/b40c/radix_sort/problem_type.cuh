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

#include <b40c/partition/problem_type.cuh>
#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace radix_sort {


/**
 * Type of radix sorting problem (i.e., data types to sort)
 *
 * Derives from partition::KeyType
 */
template <
	typename _KeyType,
	typename _ValueType,
	typename _SizeT>
struct ProblemType :
	partition::ProblemType<
		typename KeyTraits<_KeyType>::ConvertedKeyType,		// converted (unsigned) key type
		_ValueType,
		_SizeT>
{
	// The original type of data we are operating upon
	typedef _KeyType OriginalKeyType;

	// The key traits that describe any pre/post processing
	typedef KeyTraits<_KeyType> KeyTraits;
};
		

}// namespace radix_sort
}// namespace b40c

