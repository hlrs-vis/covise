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
 * Tile-processing functionality for radix sort upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/partition/upsweep/tile.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {


/**
 * Tile
 *
 * Derives from partition::upsweep::Tile
 */
template <
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE,
	typename KernelPolicy>
struct Tile :
	partition::upsweep::Tile<
		LOG_LOADS_PER_TILE,
		LOG_LOAD_VEC_SIZE,
		KernelPolicy,
		Tile<LOG_LOADS_PER_TILE, LOG_LOAD_VEC_SIZE, KernelPolicy> >					// This class
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 		KeyType;
	typedef typename KernelPolicy::SizeT 		SizeT;


	//---------------------------------------------------------------------
	// Derived Interface
	//---------------------------------------------------------------------

	/**
	 * Returns the bin into which the specified key is to be placed
	 */
	template <typename Cta>
	__device__ __forceinline__ int DecodeBin(KeyType key, Cta *cta)
	{
		int bin;
		ExtractKeyBits<
			KeyType,
			KernelPolicy::CURRENT_BIT,
			KernelPolicy::LOG_BINS>::Extract(bin, key);
		return bin;
	}


	/**
	 * Returns whether or not the key is valid.
	 *
	 * Can be overloaded.
	 */
	template <int LOAD, int VEC>
	__device__ __forceinline__ bool IsValid()
	{
		return true;
	}


	/**
	 * Loads keys into the tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		Cta *cta,
		SizeT cta_offset)
	{
		// Read tile of keys
		util::io::LoadTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::template LoadValid<
				KeyType,
				KernelPolicy::PreprocessTraits::Preprocess>(
					(KeyType (*)[Tile::LOAD_VEC_SIZE]) this->keys,
					cta->d_in_keys,
					cta_offset);
	}

	/**
	 * Stores keys from the tile (not necessary)
	 */
	template <typename Cta>
	__device__ __forceinline__ void StoreKeys(
		Cta *cta,
		SizeT cta_offset) {}
};



} // namespace upsweep
} // namespace radix_sort
} // namespace b40c
