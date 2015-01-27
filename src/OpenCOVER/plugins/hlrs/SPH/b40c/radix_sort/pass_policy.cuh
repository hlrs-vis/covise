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
 * Policy for a specific digit place pass
 ******************************************************************************/

#pragma once

namespace b40c {
namespace radix_sort {


/**
 * Policy for a specific digit place pass
 */
template <
	int 			_CURRENT_PASS,
	int 			_CURRENT_BIT,
	typename 		_PreprocessTraits,
	typename 		_PostprocessTraits>
struct PassPolicy
{
	typedef _PreprocessTraits			PreprocessTraits;		// Key pre-processing actions
	typedef _PostprocessTraits			PostprocessTraits;		// Key post-processing actions

	enum {
		CURRENT_PASS						= _CURRENT_PASS,
		CURRENT_BIT							= _CURRENT_BIT,
	};
};
		

}// namespace radix_sort
}// namespace b40c

