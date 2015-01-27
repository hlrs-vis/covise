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
 * Kernel utilities for gathering data
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace io {




/**
 * Gather a tile of data items using the corresponding tile of gather_offsets
 *
 * Uses vec-1 loads.
 */
template <
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE,
	int ACTIVE_THREADS,								// Active threads that will be loading
	ld::CacheModifier CACHE_MODIFIER>				// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
struct GatherTile
{
	static const int LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE;
	static const int LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	// Iterate next vec
	template <int LOAD, int VEC, int dummy = 0>
	struct Iterate
	{
		// predicated on valid
		template <typename T, void Transform(T&, bool), typename Flag>
		static __device__ __forceinline__ void Invoke(
			T *src,
			T dest[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE])
		{
			T *d_src = src + (LOAD * LOAD_VEC_SIZE * ACTIVE_THREADS) + (threadIdx.x << LOG_LOAD_VEC_SIZE) + VEC;

			if (valid_flags[LOAD][VEC]) {
				ModifiedLoad<CACHE_MODIFIER>::Ld(dest[LOAD][VEC], d_src);
				Transform(dest[LOAD][VEC], true);
			} else {
				Transform(dest[LOAD][VEC], false);
			}

			Iterate<LOAD, VEC + 1>::template Invoke<T, Transform, Flag>(
				src, dest, valid_flags);
		}
	};

	// Iterate next load
	template <int LOAD, int dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// predicated on valid
		template <typename T, void Transform(T&, bool), typename Flag>
		static __device__ __forceinline__ void Invoke(
			T *src,
			T dest[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE])
		{
			Iterate<LOAD + 1, 0>::template Invoke<T, Transform, Flag>(
				src, dest, valid_flags);
		}
	};

	// Terminate
	template <int dummy>
	struct Iterate<LOADS_PER_TILE, 0, dummy>
	{
		// predicated on valid
		template <typename T, void Transform(T&, bool), typename Flag>
		static __device__ __forceinline__ void Invoke(
			T *src,
			T dest[][LOAD_VEC_SIZE],
			Flag valid_flags[][LOAD_VEC_SIZE]) {}
	};

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Gather to destination with transform, predicated on the valid flag
	 */
	template <
		typename T,
		void Transform(T&, bool), 							// Assignment function to transform the loaded value
		typename Flag>
	static __device__ __forceinline__ void Gather(
		T *src,
		T dest[][LOAD_VEC_SIZE],
		Flag valid_flags[][LOAD_VEC_SIZE])
	{
		Iterate<0, 0>::template Invoke<T, Transform>(
			src, dest, valid_flags);
	}

	/**
	 * Gather to destination predicated on the valid flag
	 */
	template <typename T, typename Flag>
	static __device__ __forceinline__ void Gather(
		T *src,
		T dest[][LOAD_VEC_SIZE],
		Flag valid_flags[][LOAD_VEC_SIZE])
	{
		Gather<T, NopLdTransform<T>, Flag>(
			src, dest, valid_flags);
	}

};



} // namespace io
} // namespace util
} // namespace b40c

