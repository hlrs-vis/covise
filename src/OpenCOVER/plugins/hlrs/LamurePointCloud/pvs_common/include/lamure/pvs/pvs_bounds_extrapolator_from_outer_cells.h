// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_BOUNDS_EXTRAPOLATOR_FROM_OUTER_CELLS_H
#define LAMURE_PVS_BOUNDS_EXTRAPOLATOR_FROM_OUTER_CELLS_H

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/pvs_bounds_extrapolator.h"

namespace lamure
{
namespace pvs
{

class pvs_bounds_extrapolator_from_outer_cells : pvs_bounds_extrapolator
{
public:
	pvs_bounds_extrapolator_from_outer_cells();
	~pvs_bounds_extrapolator_from_outer_cells();

	virtual grid_bounding* extrapolate_from_grid(const grid* input_grid) const;

protected:
	void collect_visibility_in_direction(const grid* input_grid, 
										grid_bounding* output_grid, 
										const size_t& output_cell_index, 
										const scm::math::vec3d& direction, 
										const scm::math::vec3d& smallest_cell_size, 
										const size_t& num_smallest_cells_in_axis_x, 
										const size_t& num_smallest_cells_in_axis_y, 
										const size_t& num_smallest_cells_in_axis_z) const;

	void merge_visibilities_within_bounding_grid(const grid* input_grid, 
												grid_bounding* output_grid, 
												const size_t& output_cell_index, 
												const size_t& input_cell_index_one, 
												const size_t& input_cell_index_two) const;

	void merge_visibilities_within_bounding_grid(const grid* input_grid, 
												grid_bounding* output_grid, 
												const size_t& output_cell_index, 
												const size_t& input_cell_index_one, 
												const size_t& input_cell_index_two,
												const size_t& input_cell_index_three) const;
};

}
}

#endif
