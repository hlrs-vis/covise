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
 * Kernel utilities for initializing 2D arrays (tiles)
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {
namespace io {


/**
 * Initialize a tile of items
 */
template <
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE>
struct InitializeTile
{
	static const int LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE;
	static const int LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE;

	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	// Iterate over vec-elements
	template <int LOAD, int VEC>
	struct Iterate
	{
		template <typename T, typename S>
		static __device__ __forceinline__ void Copy(
			T target[][LOAD_VEC_SIZE],
			S source[][LOAD_VEC_SIZE])
		{
			target[LOAD][VEC] = source[LOAD][VEC];
			Iterate<LOAD, VEC + 1>::Copy(target, source);
		}

		template <typename T, typename S>
		static __device__ __forceinline__ void Init(
			T target[][LOAD_VEC_SIZE],
			S datum)
		{
			target[LOAD][VEC] = datum;
			Iterate<LOAD, VEC + 1>::Init(target, datum);
		}

		template <typename T, typename S, typename TransformOp>
		static __device__ __forceinline__ void Transform(
			T target[][LOAD_VEC_SIZE],
			S source[][LOAD_VEC_SIZE],
			TransformOp transform_op)
		{
			target[LOAD][VEC] = transform_op(source[LOAD][VEC]);
			Iterate<LOAD, VEC + 1>::Transform(target, source, transform_op);
		}

		template <typename SoaT, typename SoaS, typename TransformOp>
		static __device__ __forceinline__ void Transform(
			SoaT target_soa,
			SoaS source_soa,
			TransformOp transform_op)
		{
			SoaS source;
			source_soa.Get(source, LOAD, VEC);
			SoaT target = transform_op(source);
			target_soa.Set(target, LOAD, VEC);
			Iterate<LOAD, VEC + 1>::Transform(target, source, transform_op);
		}
	};

	// Iterate over loads
	template <int LOAD>
	struct Iterate<LOAD, LOAD_VEC_SIZE>
	{
		template <typename T, typename S>
		static __device__ __forceinline__ void Copy(
			T target[][LOAD_VEC_SIZE],
			S source[][LOAD_VEC_SIZE])
		{
			Iterate<LOAD + 1, 0>::Copy(target, source);
		}

		template <typename T, typename S>
		static __device__ __forceinline__ void Init(
			T target[][LOAD_VEC_SIZE],
			S datum)
		{
			Iterate<LOAD + 1, 0>::Init(target, datum);
		}

		template <typename T, typename S, typename TransformOp>
		static __device__ __forceinline__ void Transform(
			T target[][LOAD_VEC_SIZE],
			S source[][LOAD_VEC_SIZE],
			TransformOp transform_op)
		{
			Iterate<LOAD + 1, 0>::Transform(target, source, transform_op);
		}

		template <typename SoaT, typename SoaS, typename TransformOp>
		static __device__ __forceinline__ void Transform(
			SoaT target_soa,
			SoaS source_soa,
			TransformOp transform_op)
		{
			Iterate<LOAD + 1, 0>::Transform(target_soa, source_soa, transform_op);
		}
	};

	// Terminate
	template <int VEC>
	struct Iterate<LOADS_PER_TILE, VEC>
	{
		template <typename T, typename S>
		static __device__ __forceinline__ void Copy(
			T target[][LOAD_VEC_SIZE],
			S source[][LOAD_VEC_SIZE]) {}

		template <typename T, typename S>
		static __device__ __forceinline__ void Init(
			T target[][LOAD_VEC_SIZE],
			S datum) {}

		template <typename T, typename S, typename TransformOp>
		static __device__ __forceinline__ void Transform(
			T target[][LOAD_VEC_SIZE],
			S source[][LOAD_VEC_SIZE],
			TransformOp transform_op) {}

		template <typename SoaT, typename SoaS, typename TransformOp>
		static __device__ __forceinline__ void Transform(
			SoaT target_soa,
			SoaS source_soa,
			TransformOp transform_op) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Copy source to target
	 */
	template <typename T, typename S>
	static __device__ __forceinline__ void Copy(
		T target[][LOAD_VEC_SIZE],
		S source[][LOAD_VEC_SIZE])
	{
		Iterate<0, 0>::Copy(target, source);
	}


	/**
	 * Initialize target with datum
	 */
	template <typename T, typename S>
	static __device__ __forceinline__ void Init(
		T target[][LOAD_VEC_SIZE],
		S datum)
	{
		Iterate<0, 0>::Init(target, datum);
	}


	/**
	 * Apply unary transform_op operator to source
	 */
	template <typename T, typename S, typename TransformOp>
	static __device__ __forceinline__ void Transform(
		T target[][LOAD_VEC_SIZE],
		S source[][LOAD_VEC_SIZE],
		TransformOp transform_op)
	{
		Iterate<0, 0>::Transform(target, source, transform_op);
	}


	/**
	 * Apply structure-of-array transform_op operator to source
	 */
	template <typename SoaT, typename SoaS, typename TransformOp>
	static __device__ __forceinline__ void Transform(
		SoaT target_soa,
		SoaS source_soa,
		TransformOp transform_op)
	{
		Iterate<0, 0>::Transform(target_soa, source_soa, transform_op);
	}
};


} // namespace io
} // namespace util
} // namespace b40c

