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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Simple reduction operators
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {

/**
 * Static operator wrapping structure.
 *
 * (N.B. due to an NVCC/cudafe 4.0 regression, we can't specify static templated
 * functions inside other types...)
 */
template <typename T, typename R = T>
struct Operators
{
	/**
	 * Empty default transform function
	 */
	static __device__ __forceinline__ void NopTransform(T &val) {}

};


/**
 * Default equality functor
 */
template <typename T>
struct Equality
{
	__host__ __device__ __forceinline__ bool operator()(const T &a, const T &b)
	{
		return a == b;
	}
};


/**
 * Default sum functor
 */
template <typename T>
struct Sum
{
	// Binary reduction
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return a + b;
	}

	// Identity
	__host__ __device__ __forceinline__ T operator()()
	{
		return (T) 0;
	}
};




} // namespace util
} // namespace b40c

