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
 * Random Graph Construction Routines
 ******************************************************************************/

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>

#include <b40c/graph/builder/utils.cuh>

namespace b40c {
namespace graph {
namespace builder {

double Drand48()
{
	return double(rand()) / RAND_MAX;
}

bool Flip()
{
	return (rand() >= RAND_MAX / 2);
}

template <typename VertexId>
void ChoosePartition(
	VertexId *u,
	VertexId* v,
	VertexId step,
	double a,
	double b,
	double c,
	double d)
{
	double p;
	p = Drand48();

	if (p < a) {
		// do nothing

	} else if ((a < p) && (p < a+b)) {
		*v = *v + step;

	} else if ((a+b < p) && (p < a+b+c)) {
		*u = *u + step;

	} else if ((a+b+c < p) && (p < a+b+c+d)) {
		*u = *u + step;
		*v = *v + step;
	}
}

void VaryParams(double* a, double* b, double* c, double* d)
{
	double v, S;

	// Allow a max. of 5% variation
	v = 0.05;

	if (Flip())
		*a += *a * v * Drand48();
	else
		*a -= *a * v * Drand48();

	if (Flip())
		*b += *b * v * Drand48();
	else
		*b -= *b * v * Drand48();

	if (Flip())
		*c += *c * v * Drand48();
	else
		*c -= *c * v * Drand48();

	if (Flip())
		*d += *d * v * Drand48();
	else
		*d -= *d * v * Drand48();


	S = *a + *b + *c + *d;
	*a = *a/S;
	*b = *b/S;
	*c = *c/S;
	*d = *d/S;
}



/**
 * Builds a RMAT CSR graph by adding edges edges to nodes nodes by randomly choosing
 * a pair of nodes for each edge.  There are possibilities of loops and multiple 
 * edges between pairs of nodes.    
 * 
 * If src == -1, it is assigned a random node.  Otherwise it is verified 
 * to be in range of the constructed graph.
 * 
 * Returns 0 on success, 1 on failure.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildRmatGraph(
	SizeT nodes,
	SizeT edges,
	VertexId &src,
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	bool undirected,
	double a0,
	double b0,
	double c0)
{ 
	typedef CooEdgeTuple<VertexId, Value> EdgeTupleType;

	if ((nodes < 0) || (edges < 0)) {
		fprintf(stderr, "Invalid graph size: nodes=%d, edges=%d", nodes, edges);
		return -1;
	}

	time_t mark0 = time(NULL);
	printf("  Selecting %llu %s RMAT edges in COO format... ",
		(unsigned long long) edges, (undirected) ? "undirected" : "directed");
	fflush(stdout);

	// Construct COO graph

	VertexId directed_edges = (undirected) ? edges * 2 : edges;
	EdgeTupleType *coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * directed_edges);

	int progress = 0;
	for (SizeT i = 0; i < edges; i++) {

		double a = a0;
		double b = b0;
		double c = c0;
		double d = 1.0 - (a0 + b0 + c0);

		VertexId u 		= 1;
		VertexId v 		= 1;
		VertexId step 	= nodes / 2;

		while (step >= 1) {
			ChoosePartition(&u, &v, step, a, b, c, d);
			step /= 2;
			VaryParams(&a, &b, &c, &d);
		}

		// Create edge
		coo[i].row = u;
		coo[i].col = v;
		coo[i].val = 1;

		if (undirected) {
			// Reverse edge
			coo[edges + i].row = coo[i].col;
			coo[edges + i].col = coo[i].row;
			coo[edges + i].val = 1;
		}

		if (i > (directed_edges / 32) * (progress + 1)) {
			progress++;
			printf("%.2f%%\n", float(progress) * (100.0 / 32.0));
			fflush(stdout);
		}

	}

	time_t mark1 = time(NULL);
	printf("Done selecting (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);
	
	// Convert sorted COO to CSR
	csr_graph.template FromCoo<LOAD_VALUES>(coo, nodes, directed_edges);
	free(coo);

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
