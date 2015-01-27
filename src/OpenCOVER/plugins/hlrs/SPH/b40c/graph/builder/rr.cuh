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
 * Random-regular(ish) Graph Construction Routines
 ******************************************************************************/

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>

#include <b40c/graph/builder/utils.cuh>

namespace b40c {
namespace graph {
namespace builder {


/**
 * A random graph where each node has a guaranteed degree of random neighbors.
 * Does not meet definition of random-regular: loops, and multi-edges are 
 * possible, and in-degree is not guaranteed to be the same as out degree.   
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildRandomRegularishGraph(
	SizeT nodes,
	int degree,
	VertexId &src,
	CsrGraph<VertexId, Value, SizeT> &csr_graph)
{
	SizeT edges 				= nodes * degree;
	
	csr_graph.template FromScratch<LOAD_VALUES>(nodes, edges);

	time_t mark0 = time(NULL);
	printf("  Selecting %llu random edges in COO format... ",
		(unsigned long long) edges);
	fflush(stdout);

	SizeT total = 0;
    for (VertexId node = 0; node < nodes; node++) {
    	
    	csr_graph.row_offsets[node] = total;
    	
    	for (int edge = 0; edge < degree; edge++) {
    		
    		VertexId neighbor = RandomNode(csr_graph.nodes);
    		csr_graph.column_indices[total] = neighbor;
    		if (LOAD_VALUES) {
    			csr_graph.values[node] = 1;
    		}
    		
    		total++;
    	}
    }
    
    csr_graph.row_offsets[nodes] = total; 	// last offset is always num_entries

	time_t mark1 = time(NULL);
	printf("Done selecting (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);

	// If unspecified, assign default source.  Otherwise verify source range.
	if (src == -1) {
		// Random source
		src = RandomNode(csr_graph.nodes);
	} else if ((src < 0 ) || (src > csr_graph.nodes)) {
		fprintf(stderr, "Invalid src: %d", src);
		csr_graph.Free();
		return -1;
	}
	
	return 0;
}


} // namespace builder
} // namespace graph
} // namespace b40c
