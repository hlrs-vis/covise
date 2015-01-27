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
 * MARKET Graph Construction Routines
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
 * Reads a MARKET graph from an input-stream into a CSR sparse format
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int ReadMarketStream(
	FILE *f_in,
	CsrGraph<VertexId, Value, SizeT> &csr_graph)
{
	typedef CooEdgeTuple<VertexId, Value> EdgeTupleType;
	
	SizeT edges_read = -1;
	SizeT nodes = 0;
	SizeT edges = 0;
	EdgeTupleType *coo = NULL;		// read in COO format
	
	time_t mark0 = time(NULL);
	printf("  Parsing MARKET COO format ");
	fflush(stdout);

	char line[1024];

	while(true) {

		if (fscanf(f_in, "%[^\n]\n", line) <= 0) {
			break;
		}

		if (line[0] == '%') {

			// Comment

		} else if (edges_read == -1) {

			// Problem description
			long long ll_nodes_x, ll_nodes_y, ll_edges;
			if (sscanf(line, "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges) != 3) {
				fprintf(stderr, "Error parsing MARKET graph: invalid problem description\n");
				return -1;
			}

			if (ll_nodes_x != ll_nodes_y) {
				fprintf(stderr, "Error parsing MARKET graph: not square (%lld, %lld)\n", ll_nodes_x, ll_nodes_y);
				return -1;
			}

			nodes = ll_nodes_x;
			edges = ll_edges;

			printf(" (%lld nodes, %lld directed edges)... ",
				(unsigned long long) ll_nodes_x, (unsigned long long) ll_edges);
			fflush(stdout);
			
			// Allocate coo graph
			coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * edges);

			edges_read++;

		} else {

			// Edge description (v -> w)
			if (!coo) {
				fprintf(stderr, "Error parsing MARKET graph: invalid format\n");
				return -1;
			}			
			if (edges_read >= edges) {
				fprintf(stderr, "Error parsing MARKET graph: encountered more than %d edges\n", edges);
				if (coo) free(coo);
				return -1;
			}

			long long ll_row, ll_col;
			if (sscanf(line, "%lld %lld", &ll_col, &ll_row) != 2) {
				fprintf(stderr, "Error parsing MARKET graph: badly formed edge\n", edges);
				if (coo) free(coo);
				return -1;
			}

			coo[edges_read].row = ll_row - 1;	// zero-based array
			coo[edges_read].col = ll_col - 1;	// zero-based array

			edges_read++;
		}
	}
	
	if (coo == NULL) {
		fprintf(stderr, "No graph found\n");
		return -1;
	}

	if (edges_read != edges) {
		fprintf(stderr, "Error parsing MARKET graph: only %d/%d edges read\n", edges_read, edges);
		if (coo) free(coo);
		return -1;
	}
	
	time_t mark1 = time(NULL);
	printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);
	
	// Convert sorted COO to CSR
	csr_graph.template FromCoo<LOAD_VALUES>(coo, nodes, edges);
	free(coo);

	fflush(stdout);
	
	return 0;
}


/**
 * Loads a MARKET-formatted CSR graph from the specified file.  If
 * dimacs_filename == NULL, then it is loaded from stdin.
 * 
 * If src == -1, it is assigned a random node.  Otherwise it is verified 
 * to be in range of the constructed graph.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildMarketGraph(
	char *dimacs_filename, 
	VertexId &src,
	CsrGraph<VertexId, Value, SizeT> &csr_graph)
{ 
	if (dimacs_filename == NULL) {

		// Read from stdin
		printf("Reading from stdin:\n");
		if (ReadMarketStream<LOAD_VALUES>(stdin, csr_graph) != 0) {
			return -1;
		}

	} else {
	
		// Read from file
		FILE *f_in = fopen(dimacs_filename, "r");
		if (f_in) {
			printf("Reading from %s:\n", dimacs_filename);
			if (ReadMarketStream<LOAD_VALUES>(f_in, csr_graph) != 0) {
				fclose(f_in);
				return -1;
			}
			fclose(f_in);
		} else {
			perror("Unable to open file");
			return -1;
		}
	}
	
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
