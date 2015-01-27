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
 * Serial tuple reduction over structure-of-array types.
 ******************************************************************************/

#pragma once

#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace util {
namespace reduction {
namespace soa {


/**
 * Have each thread perform a serial reduction over its specified SOA segment
 */
template <int NUM_ELEMENTS>					// Length of SOA array segment to reduce
struct SerialSoaReduce
{
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Next SOA tuple
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		// Reduce
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Reduce(
			RakingSoa raking_partials,
			Tuple exclusive_partial,
			ReductionOp reduction_op)
		{
			// Load current partial
			Tuple current_partial;
			raking_partials.Get(current_partial, COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = reduction_op(exclusive_partial, current_partial);

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Reduce(
				raking_partials, inclusive_partial, reduction_op);
		}

		// Reduce 2D
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Reduce(
			RakingSoa raking_partials,
			Tuple exclusive_partial,
			int row,
			ReductionOp reduction_op)
		{
			// Load current partial
			Tuple current_partial;
			raking_partials.Get(current_partial, row, COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = reduction_op(exclusive_partial, current_partial);

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Reduce(
				raking_partials, inclusive_partial, row, reduction_op);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		// Reduce
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Reduce(
			RakingSoa raking_partials,
			Tuple exclusive_partial,
			ReductionOp reduction_op)
		{
			return exclusive_partial;
		}

		// Reduce 2D
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Reduce(
			RakingSoa raking_partials,
			Tuple exclusive_partial,
			int row,
			ReductionOp reduction_op)
		{
			return exclusive_partial;
		}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Reduce a structure-of-array RakingSoa into a single Tuple "slice"
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple Reduce(
		Tuple &retval,
		RakingSoa raking_partials,
		ReductionOp reduction_op)
	{
		// Get first partial
		Tuple current_partial;
		raking_partials.Get(current_partial, 0);

		retval = Iterate<1, NUM_ELEMENTS>::Reduce(
			raking_partials, current_partial, reduction_op);
		return retval;
	}

	/**
	 * Reduce a structure-of-array RakingSoa into a single Tuple "slice", seeded
	 * with exclusive_partial
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple SeedReduce(
		RakingSoa raking_partials,
		Tuple exclusive_partial,
		ReductionOp reduction_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Reduce(
			raking_partials, exclusive_partial, reduction_op);
	}


	/**
	 * Reduce one row of a 2D structure-of-array RakingSoa into a single Tuple "slice"
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple Reduce(
		Tuple &retval,
		RakingSoa raking_partials,
		int row,
		ReductionOp reduction_op)
	{
		// Get first partial
		Tuple current_partial;
		raking_partials.Get(current_partial, row, 0);

		retval = Iterate<1, NUM_ELEMENTS>::Reduce(
			raking_partials, current_partial, row, reduction_op);
		return retval;
	}


	/**
	 * Reduce one row of a 2D structure-of-array RakingSoa into a single Tuple "slice", seeded
	 * with exclusive_partial
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple SeedReduce(
		RakingSoa raking_partials,
		Tuple exclusive_partial,
		int row,
		ReductionOp reduction_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Reduce(
			raking_partials, exclusive_partial, row, reduction_op);
	}
};


} // namespace soa
} // namespace reduction
} // namespace util
} // namespace b40c

