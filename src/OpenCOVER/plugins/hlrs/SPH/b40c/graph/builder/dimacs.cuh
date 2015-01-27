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
 * DIMACS Graph Construction Routines
 ******************************************************************************/

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>

#include <vector>
#include <string>

#include <b40c/util/basic_utils.cuh>
#include <b40c/graph/builder/utils.cuh>

namespace b40c {
namespace graph {
namespace builder {

/**
 * Reads a DIMACS graph from an input-stream into a CSR sparse format 
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int ReadDimacsStream(
	std::vector<FILE *> files,
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	bool undirected)
{
	typedef typename util::If<LOAD_VALUES, Value, util::NullType>::Type TupleValue;
	typedef CooEdgeTuple<VertexId, TupleValue> EdgeTupleType;
	
	SizeT edges_read = 0;
	SizeT nodes = 0;
	SizeT edges = 0;
	SizeT directed_edges = 0;
	EdgeTupleType *coo = NULL;		// read in COO format
	
	time_t mark0 = time(NULL);
	printf("  Parsing DIMACS COO format ");
	fflush(stdout);

	char line[1024];
	char problem_type[1024];

	bool ordered_rows = true;

	for (int file = 0; file < files.size(); file++) {

		bool parsed = false;
		while(!parsed) {

			if (fscanf(files[file], "%[^\n]\n", line) <= 0) {
				break;
			}

			switch (line[0]) {
			case 'p':
			{
				// Problem description (nodes is nodes, edges is edges)
				long long ll_nodes, ll_edges;
				sscanf(line, "p %s %lld %lld", problem_type, &ll_nodes, &ll_edges);
				if (nodes && (nodes != ll_nodes)) {
					fprintf(stderr, "Error: splice files do not name the same number of vertices\n");
					return -1;
				} else {
					nodes = ll_nodes;
				}
				edges += ll_edges;
				parsed = true;

				break;
			}
			default:
				// read remainder of line
				break;
			}
		}
	}

	directed_edges = (undirected) ? edges * 2 : edges;
	if (!directed_edges) {
		fprintf(stderr, "No graph found\n");
		return -1;
	}

	printf(" (%lld vertices, %lld directed edges)... ",
		(unsigned long long) nodes, (unsigned long long) directed_edges);
	fflush(stdout);

	// Allocate coo graph
	coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * directed_edges);

	// Vector of latest tuples
	std::vector<EdgeTupleType> tuples(files.size(), EdgeTupleType(-1, 0, 0));

	int progress = 0;

	// Splice in ordered vertices
	while(true) {

		// Pick the smallest edge from the file set and add it to the COO edge list
		int smallest = -1;
		for (int i = 0; i < files.size(); i++) {
			
			// Read a tuple from this file if necessary
			while ((tuples[i].row < 0) && (fscanf(files[i], "%[^\n]\n", line) > 0)) {

				switch (line[0]) {
				case 'a':
				{
					// Edge description (v -> w) with value val
					if (!coo) {
						fprintf(stderr, "Error parsing DIMACS graph: invalid format\n");
						return -1;
					}
					if (edges_read >= directed_edges) {
						fprintf(stderr, "Error parsing DIMACS graph: encountered more than %d edges\n", directed_edges);
						if (coo) free(coo);
						return -1;
					}

					long long ll_row, ll_col, ll_val;
					sscanf(line, "a %lld %lld %lld", &ll_row, &ll_col, &ll_val);

					tuples[i] = EdgeTupleType(
						ll_row - 1,	// zero-based array
						ll_col - 1,	// zero-based array
						ll_val);

					if (undirected) {
						// Go ahead and insert reverse edge
						coo[edges_read] = EdgeTupleType(
							ll_col - 1,	// zero-based array
							ll_row - 1,	// zero-based array
							ll_val);

						ordered_rows = false;
						edges_read++;
					}

					if (edges_read > (directed_edges / 32) * (progress + 1)) {
						progress++;
						printf("%.2f%%\n", float(progress) * (100.0 / 32.0));
						fflush(stdout);
					}

					break;
				}

				default:
					// read remainder of line
					break;
				}
			}

			// Compare this tuple against the smallest one so far
			if ((tuples[i].row >= 0) && ((smallest < 0) || (tuples[i].row < tuples[smallest].row))) {
				smallest = i;
			}
		}

		// Insert smallest edge from the splice files (or quit if none)
		if (smallest < 0) {
			break;
		} else {
			if (edges_read && (tuples[smallest].row < coo[edges_read - 1].row)) {
				ordered_rows = false;
			}
			coo[edges_read] = tuples[smallest];
			tuples[smallest].row = -1;
			smallest = -1;
			edges_read++;
		}
	}

	if (edges_read != directed_edges) {
		fprintf(stderr, "Error parsing DIMACS graph: only %d/%d edges read\n", edges_read, directed_edges);
		if (coo) free(coo);
		return -1;
	}
	
	time_t mark1 = time(NULL);
	printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);

	// Convert sorted COO to CSR
	csr_graph.template FromCoo<LOAD_VALUES>(coo, nodes, directed_edges, ordered_rows);
	free(coo);

	fflush(stdout);
	return 0;
}


/**
 * Loads a DIMACS-formatted CSR graph from the specified file.  If 
 * dimacs_filename == NULL, then it is loaded from stdin.
 * 
 * If src == -1, it is assigned a random node.  Otherwise it is verified 
 * to be in range of the constructed graph.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildDimacsGraph(
	char *dimacs_filename, 
	VertexId &src,
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	bool undirected,
	int splice)
{ 
	int retval = 0;

	std::vector<FILE*> files;
	if (dimacs_filename == NULL) {

		// Read from stdin
		printf("Reading from stdin:\n");
		files.push_back(stdin);
		if (ReadDimacsStream<LOAD_VALUES>(files, csr_graph, undirected) != 0) {
			retval = -1;
		}
	} else {
	
		// Read from file(s)
		FILE *f_in;
		if (splice) {
			for (int i = 0; i < splice; i++) {
				std::stringstream formatter;
				formatter << dimacs_filename << "." << i;
				if ((f_in = fopen(formatter.str().c_str(), "r")) == NULL) {
					break;
				}
				files.push_back(f_in);
				printf("Opened %s\n", formatter.str().c_str());
			}
		} else {
			if ((f_in = fopen(dimacs_filename, "r")) != NULL) {
				files.push_back(f_in);
				printf("Opened %s:\n", dimacs_filename);
			}
		}
		if (files.size()) {
			retval = ReadDimacsStream<LOAD_VALUES>(files, csr_graph, undirected);
			for (int i = 0; i < files.size(); i++) {
				if (files[i]) fclose(files[i]);
			}
		} else {
			perror("Unable to open file");
			retval = -1;
		}
	}

	if (!retval) {
		// If unspecified, assign default source.  Otherwise verify source range.
		if (src == -1) {
			// Random source
			src = RandomNode(csr_graph.nodes);
		} else if ((src < 0 ) || (src > csr_graph.nodes)) {
			fprintf(stderr, "Invalid src: %d", src);
			csr_graph.Free();
			retval = -1;
		}
	}
	
	return retval;
}

} // namespace builder
} // namespace graph
} // namespace b40c
