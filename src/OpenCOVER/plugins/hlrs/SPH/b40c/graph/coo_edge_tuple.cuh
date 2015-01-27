/******************************************************************************
 * Copyright 2010 Duane Merrill
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
 ******************************************************************************/


/******************************************************************************
 * Simple COO sparse graph data structure
 ******************************************************************************/

#pragma once

#include <time.h>
#include <stdio.h>

#include <algorithm>

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/error_utils.cuh>

namespace b40c {
namespace graph {


/**
 * COO sparse format edge.  (A COO graph is just a list/array/vector of these.)
 */
template<typename VertexId, typename Value>
struct CooEdgeTuple {
	VertexId row;
	VertexId col;
	Value val;

	CooEdgeTuple(VertexId row, VertexId col, Value val) : row(row), col(col), val(val) {}

	void Val(Value &value)
	{
		value = val;
	}
};


template<typename VertexId>
struct CooEdgeTuple<VertexId, util::NullType> {
	VertexId row;
	VertexId col;

	template <typename Value>
	CooEdgeTuple(VertexId row, VertexId col, Value val) : row(row), col(col) {}

	template <typename Value>
	void Val(Value &value) {}
};


/**
 * Comparator for sorting COO sparse format edges
 */
template<typename CooEdgeTuple>
bool DimacsTupleCompare (
	CooEdgeTuple elem1,
	CooEdgeTuple elem2)
{
	if (elem1.row < elem2.row) {
		// Sort edges by source node (to make rows)
		return true;
/*
	} else if ((elem1.row == elem2.row) && (elem1.col < elem2.col)) {
		// Sort edgelists as well for coherence
		return true;
*/
	} 
	
	return false;
}


} // namespace graph
} // namespace b40c
