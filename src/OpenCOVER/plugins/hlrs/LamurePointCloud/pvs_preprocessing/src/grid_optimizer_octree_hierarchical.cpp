// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/grid_optimizer_octree_hierarchical.h"
#include "lamure/pvs/grid_octree_hierarchical.h"
#include "lamure/pvs/grid_octree_hierarchical_node.h"

namespace lamure
{
namespace pvs
{

void grid_optimizer_octree_hierarchical::
optimize_grid(grid* input_grid, const float& equality_threshold)
{
	grid_octree_hierarchical* oct_hier_grid = (grid_octree_hierarchical*)input_grid;

	unsigned short num_allowed_unequal_elements = 8 - std::round(8.0f * equality_threshold); 
	oct_hier_grid->combine_visibility(num_allowed_unequal_elements);
}

}
}