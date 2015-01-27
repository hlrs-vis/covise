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
 *  Storage wrapper for multi-pass stream transformations that require a
 *  secondary problem storage array to stream results back and forth from.
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace util {


/**
 * Storage wrapper for multi-pass stream transformations that require a
 * secondary problem storage array to stream results back and forth from.
 * 
 * This wrapper provides maximum flexibility for re-using device allocations
 * for subsequent transformations.  As such, it is the caller's responsibility
 * to free any non-NULL storage arrays when no longer needed.
 * 
 * Many multi-pass stream computations require two problem storage arrays: one
 * for reading in from, the other for writing out to.  (And their roles can
 * be reversed for each subsequent pass.) This structure tracks two pairs of
 * device vectors (a keys pair and a values pair), and a "selector" member to
 * index which vector in each pair is "currently valid".  I.e., the valid data
 * is accessible by:
 * 
 * 		<storage>.d_keys[<storage>.selector];
 * 
 * Many of the B40C primitives are templated upon the PingPongStorage type: they
 * are compiled differently depending upon whether the declared type contains
 * keys-only versus key-value pairs (i.e., whether ValueType is util::NullType
 * or some real type).
 *
 * Declaring keys-only storage wrapper:
 *
 * 		PingPongStorage<KeyType> key_storage;
 *
 * Declaring key-value storage wrapper:
 *
 * 		PingPongStorage<KeyType, ValueType> key_value_storage;
 *
 */
template <
	typename _KeyType,
	typename _ValueType = util::NullType>
struct PingPongStorage
{
	typedef _KeyType	KeyType;
	typedef _ValueType 	ValueType;

	// Pair of device vector pointers for keys
	KeyType* d_keys[2];
	
	// Pair of device vector pointers for values
	ValueType* d_values[2];

	// Selector into the pair of device vector pointers indicating valid 
	// sorting elements (i.e., where the results are)
	int selector;

	// Constructor
	PingPongStorage()
	{
		selector = 0;
		d_keys[0] = NULL;
		d_keys[1] = NULL;
		d_values[0] = NULL;
		d_values[1] = NULL;
	}

	// Constructor
	PingPongStorage(
		KeyType* keys)
	{
		selector = 0;
		d_keys[0] = keys;
		d_keys[1] = NULL;
		d_values[0] = NULL;
		d_values[1] = NULL;
	}

	// Constructor
	PingPongStorage(
		KeyType* keys,
		ValueType* values)
	{
		selector = 0;
		d_keys[0] = keys;
		d_keys[1] = NULL;
		d_values[0] = values;
		d_values[1] = NULL;
	}

	// Constructor
	PingPongStorage(
		KeyType* keys0,
		KeyType* keys1,
		ValueType* values0,
		ValueType* values1)
	{
		selector = 0;
		d_keys[0] = keys0;
		d_keys[1] = keys1;
		d_values[0] = values0;
		d_values[1] = values1;
	}
};

} // namespace util
} // namespace b40c

