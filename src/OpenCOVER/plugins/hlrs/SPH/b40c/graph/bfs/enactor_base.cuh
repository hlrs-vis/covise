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
 * Base BFS Search Enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/error_utils.cuh>

namespace b40c {
namespace graph {
namespace bfs {



/**
 * Base class for breadth-first-search enactors.
 * 
 * A BFS search iteratively expands outwards from the given source node.  At 
 * each iteration, the algorithm discovers unvisited nodes that are adjacent 
 * to the nodes discovered by the previous iteration.  The first iteration 
 * discovers the source node. 
 */
class EnactorBase
{
protected:	

	//Device properties
	util::CudaProperties cuda_props;
	
	// Queue size counters and accompanying functionality
	util::CtaWorkProgressLifetime work_progress;

public:

	// Allows display to stdout of search details
	bool DEBUG;

protected: 	

	/**
	 * Constructor.
	 */
	EnactorBase(bool DEBUG) : DEBUG(DEBUG)
	{
		// Setup work progress (only needs doing once since we maintain
		// it in our kernel code)
		work_progress.Setup();
	}


	/**
	 * Utility function: Returns the default maximum number of threadblocks
	 * this enactor class can launch.
	 */
	int MaxGridSize(int cta_occupancy, int max_grid_size = 0)
	{
		if (max_grid_size <= 0) {
			// No override: Fully populate all SMs
			max_grid_size = this->cuda_props.device_props.multiProcessorCount * cta_occupancy;
		}

		return max_grid_size;
	}


	/**
	 * Utility method to display the contents of a device array
	 */
	template <typename T>
	void DisplayDeviceResults(
		T *d_data,
		size_t num_elements)
	{
		// Allocate array on host and copy back
		T *h_data = (T*) malloc(num_elements * sizeof(T));
		cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

		// Display data
		for (int i = 0; i < num_elements; i++) {
			PrintValue(h_data[i]);
			printf(", ");
		}
		printf("\n\n");

		// Cleanup
		if (h_data) free(h_data);
	}
};


} // namespace bfs
} // namespace graph
} // namespace b40c
