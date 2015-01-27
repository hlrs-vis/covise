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
 * Serial tuple scan over structure-of-array types.
 ******************************************************************************/

#pragma once

#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace util {
namespace scan {
namespace soa {


/**
 * Have each thread perform a serial scan over its specified SOA segment
 */
template <
	int NUM_ELEMENTS,				// Length of SOA array segment to scan
	bool EXCLUSIVE = true>			// Whether or not this is an exclusive scan
struct SerialSoaScan
{
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		// Scan
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Scan(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial,
			ReductionOp scan_op)
		{
			// Load current partial
			Tuple current_partial;
			raking_partials.Get(current_partial, COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = scan_op(exclusive_partial, current_partial);

			if (EXCLUSIVE) {
				// Store exclusive partial
				raking_results.Set(exclusive_partial, COUNT);
			} else {
				// Store inclusive partial
				raking_results.Set(inclusive_partial, COUNT);
			}

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Scan(
				raking_partials, raking_results, inclusive_partial, scan_op);
		}

		// Scan 2D
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Scan(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial,
			int row,
			ReductionOp scan_op)
		{
			// Load current partial
			Tuple current_partial;
			raking_partials.Get(current_partial, row, COUNT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = scan_op(exclusive_partial, current_partial);

			if (EXCLUSIVE) {
				// Store exclusive partial
				raking_results.Set(exclusive_partial, row, COUNT);
			} else {
				// Store inclusive partial
				raking_results.Set(inclusive_partial, row, COUNT);
			}

			// Recurse
			return Iterate<COUNT + 1, TOTAL>::Scan(
				raking_partials, raking_results, inclusive_partial, row, scan_op);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<TOTAL, TOTAL>
	{
		// Scan
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Scan(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial,
			ReductionOp scan_op)
		{
			return exclusive_partial;
		}

		// Scan 2D
		template <
			typename Tuple,
			typename RakingSoa,
			typename ReductionOp>
		static __host__ __device__ __forceinline__ Tuple Scan(
			RakingSoa raking_partials,
			RakingSoa raking_results,
			Tuple exclusive_partial,
			int row,
			ReductionOp scan_op)
		{
			return exclusive_partial;
		}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Scan a 2D structure-of-array RakingSoa, seeded  with exclusive_partial.
	 * The tuple returned is the inclusive total.
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple Scan(
		RakingSoa raking_partials,			// Scan input/output
		Tuple exclusive_partial,			// Exclusive partial to seed with
		ReductionOp scan_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Scan(
			raking_partials, raking_partials, exclusive_partial, scan_op);
	}


	/**
	 * Scan a 2D structure-of-array RakingSoa, seeded  with exclusive_partial.
	 * The tuple returned is the inclusive total.
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple Scan(
		RakingSoa raking_partials,			// Scan input
		RakingSoa raking_results,			// Scan output
		Tuple exclusive_partial,			// Exclusive partial to seed with
		ReductionOp scan_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Scan(
			raking_partials, raking_results, exclusive_partial, scan_op);
	}


	/**
	 * Scan one row of a 2D structure-of-array RakingSoa, seeded
	 * with exclusive_partial.  The tuple returned is the inclusive total.
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple Scan(
		RakingSoa raking_partials,			// Scan input/output
		Tuple exclusive_partial,			// Exclusive partial to seed with
		int row,
		ReductionOp scan_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Scan(
			raking_partials, raking_partials, exclusive_partial, row, scan_op);
	}


	/**
	 * Scan one row of a 2D structure-of-array RakingSoa, seeded
	 * with exclusive_partial.  The tuple returned is the inclusive total.
	 */
	template <
		typename Tuple,
		typename RakingSoa,
		typename ReductionOp>
	static __host__ __device__ __forceinline__ Tuple Scan(
		RakingSoa raking_partials,			// Scan input
		RakingSoa raking_results,			// Scan output
		Tuple exclusive_partial,			// Exclusive partial to seed with
		int row,
		ReductionOp scan_op)
	{
		return Iterate<0, NUM_ELEMENTS>::Scan(
			raking_partials, raking_results, exclusive_partial, row, scan_op);
	}
};

} // namespace soa
} // namespace scan
} // namespace util
} // namespace b40c

