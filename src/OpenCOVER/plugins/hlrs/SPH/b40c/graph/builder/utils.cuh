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
 * General graph-building utility routines
 ******************************************************************************/

#pragma once

#include <time.h>
#include <stdio.h>

#include <algorithm>

#include <b40c/util/error_utils.cuh>
#include <b40c/util/random_bits.cuh>

#include <b40c/graph/coo_edge_tuple.cuh>
#include <b40c/graph/csr_graph.cuh>

namespace b40c {
namespace graph {
namespace builder {


/**
 * Returns a random node-ID in the range of [0, num_nodes) 
 */
template<typename SizeT>
SizeT RandomNode(SizeT num_nodes) {
	SizeT node_id;
	util::RandomBits(node_id);
	if (node_id < 0) node_id *= -1;
	return node_id % num_nodes;
}


} // namespace builder
} // namespace graph
} // namespace b40c
