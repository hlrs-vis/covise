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
 * CTA-processing functionality for radix sort downsweep scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/partition/downsweep/cta.cuh>
#include <b40c/radix_sort/downsweep/tile.cuh>

namespace b40c {
namespace radix_sort {
namespace downsweep {


/**
 * Radix sort downsweep scan CTA
 *
 * Derives from partition::downsweep::Cta
 */
template <typename KernelPolicy>
struct Cta :
	partition::downsweep::Cta<
		KernelPolicy,
		Cta<KernelPolicy>,			// This class
		Tile>						// radix_sort::downsweep::Tile
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	// Base class type
	typedef partition::downsweep::Cta<KernelPolicy, Cta, Tile> Base;

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::ValueType 				ValueType;
	typedef typename KernelPolicy::SizeT 					SizeT;
	typedef typename KernelPolicy::SmemStorage				SmemStorage;
	typedef typename KernelPolicy::Grid::LanePartial		LanePartial;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*&d_in_keys,
		KeyType 		*&d_out_keys,
		ValueType 		*&d_in_values,
		ValueType 		*&d_out_values,
		SizeT 			*&d_spine,
		LanePartial		base_composite_counter,
		int				*raking_segment) :
			Base(
				smem_storage,
				d_in_keys,
				d_out_keys,
				d_in_values,
				d_out_values,
				d_spine,
				base_composite_counter,
				raking_segment)
	{}
};


} // namespace downsweep
} // namespace radix_sort
} // namespace b40c

