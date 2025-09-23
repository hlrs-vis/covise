// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_GRID_OPTIMIZER_OCTREE_H
#define LAMURE_PVS_GRID_OPTIMIZER_OCTREE_H

#include <lamure/pvs/pvs_preprocessing.h>
#include "lamure/pvs/grid.h"
#include "lamure/pvs/grid_octree.h"
#include "lamure/pvs/grid_octree_node.h"

namespace lamure
{
namespace pvs
{

class grid_optimizer_octree
{
public:
	// Equality threshold is the allowed difference in percent used to consider two cells as equal.
	// E.g. a value of 0.9 means the cells must contain 90% equal elements to be considered equal.
	void optimize_grid(grid* input_grid, const float& equality_threshold);

protected:
	bool check_and_optimize_node(grid_octree_node* node, grid* input_grid, const float& equality_threshold);
	bool try_collapse_node(grid_octree_node* node, grid* input_grid, const float& equality_threshold);
};

}
}

#endif
