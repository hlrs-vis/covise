// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/pvs_bounds_extrapolator_from_outer_cells.h"

namespace lamure
{
namespace pvs
{

pvs_bounds_extrapolator_from_outer_cells::
pvs_bounds_extrapolator_from_outer_cells()
{
}

pvs_bounds_extrapolator_from_outer_cells::
~pvs_bounds_extrapolator_from_outer_cells()
{
}

grid_bounding* pvs_bounds_extrapolator_from_outer_cells::
extrapolate_from_grid(const grid* input_grid) const
{
	std::vector<node_t> ids;
	for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
	{
		ids.push_back(input_grid->get_num_nodes(model_index));
	}

	grid_bounding* bounding_grid = new grid_bounding(input_grid, ids);

	if(input_grid->get_cell_count() < 1)
	{
		return nullptr;
	}

	// Find the size of the smallest cell in each axis. Required for later collection of visibility of direction.
	double smallest_cell_size_x = input_grid->get_cell_at_index(0)->get_size().x;
	double smallest_cell_size_y = input_grid->get_cell_at_index(0)->get_size().y;
	double smallest_cell_size_z = input_grid->get_cell_at_index(0)->get_size().z;

	for(size_t cell_index = 1; cell_index < input_grid->get_cell_count(); ++cell_index)
	{
		scm::math::vec3d current_cell_size = input_grid->get_cell_at_index(cell_index)->get_size();
		
		if(current_cell_size.x < smallest_cell_size_x)
		{
			smallest_cell_size_x = current_cell_size.x;
		}
		if(current_cell_size.y < smallest_cell_size_y)
		{
			smallest_cell_size_y = current_cell_size.y;
		}
		if(current_cell_size.z < smallest_cell_size_z)
		{
			smallest_cell_size_z = current_cell_size.z;
		}
	}

	size_t num_smallest_cells_in_axis_x = std::round(input_grid->get_size().x / smallest_cell_size_x);
	size_t num_smallest_cells_in_axis_y = std::round(input_grid->get_size().y / smallest_cell_size_y);
	size_t num_smallest_cells_in_axis_z = std::round(input_grid->get_size().z / smallest_cell_size_z);

	scm::math::vec3d smallest_cell_size(smallest_cell_size_x, smallest_cell_size_y, smallest_cell_size_z);

	// X-axis.
	this->collect_visibility_in_direction(input_grid, bounding_grid, 14, scm::math::vec3d(1.0, -1.0, -1.0), smallest_cell_size, 1, num_smallest_cells_in_axis_y, num_smallest_cells_in_axis_z);
	this->collect_visibility_in_direction(input_grid, bounding_grid, 12, scm::math::vec3d(-1.0, -1.0, -1.0), smallest_cell_size, 1, num_smallest_cells_in_axis_y, num_smallest_cells_in_axis_z);
	// Y-axis.
	this->collect_visibility_in_direction(input_grid, bounding_grid, 16, scm::math::vec3d(-1.0, 1.0, -1.0), smallest_cell_size, num_smallest_cells_in_axis_x, 1, num_smallest_cells_in_axis_z);
	this->collect_visibility_in_direction(input_grid, bounding_grid, 10, scm::math::vec3d(-1.0, -1.0, -1.0), smallest_cell_size, num_smallest_cells_in_axis_x, 1, num_smallest_cells_in_axis_z);
	// Z-axis.
	this->collect_visibility_in_direction(input_grid, bounding_grid, 22, scm::math::vec3d(-1.0, -1.0, 1.0), smallest_cell_size, num_smallest_cells_in_axis_x, num_smallest_cells_in_axis_y, 1);
	this->collect_visibility_in_direction(input_grid, bounding_grid, 4, scm::math::vec3d(-1.0, -1.0, -1.0), smallest_cell_size, num_smallest_cells_in_axis_x, num_smallest_cells_in_axis_y, 1);

	// Corner cells that are merged from 2 neighbouring cells.
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 1, 4, 10);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 3, 4, 12);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 5, 4, 14);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 7, 4, 16);

	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 9, 10, 12);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 11, 10, 14);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 15, 12, 16);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 17, 14, 16);

	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 19, 10, 22);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 21, 12, 22);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 23, 14, 22);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 25, 16, 22);

	// Corner cells that are merged from 3 neighbouring cells.
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 0, 1, 3, 9);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 2, 1, 5, 11);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 6, 3, 7, 15);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 8, 5, 7, 17);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 18, 9, 19, 21);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 20, 11, 19, 23);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 24, 15, 21, 25);
	this->merge_visibilities_within_bounding_grid(input_grid, bounding_grid, 26, 17, 23, 25);

	return bounding_grid;
}

void pvs_bounds_extrapolator_from_outer_cells::
collect_visibility_in_direction(const grid* input_grid, 
								grid_bounding* output_grid, 
								const size_t& output_cell_index, 
								const scm::math::vec3d& direction, 
								const scm::math::vec3d& smallest_cell_size, 
								const size_t& num_smallest_cells_in_axis_x, 
								const size_t& num_smallest_cells_in_axis_y,
								const size_t& num_smallest_cells_in_axis_z) const
{
	// Create visibility data to be joined among each outer view cell.
	std::vector<boost::dynamic_bitset<>> outer_cell_visibility;
	outer_cell_visibility.resize(input_grid->get_num_models());

	for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
	{
		outer_cell_visibility[model_index].resize(input_grid->get_num_nodes(model_index));
	}

	// Positive X axis.
	scm::math::vec3d to_bounds = input_grid->get_size() * direction * 0.5;
	scm::math::vec3d start_pos = input_grid->get_position_center() + to_bounds + (smallest_cell_size * -0.5 * direction);

	for(size_t cell_index_z = 0; cell_index_z < num_smallest_cells_in_axis_z; ++cell_index_z)
	{
		for(size_t cell_index_y = 0; cell_index_y < num_smallest_cells_in_axis_y; ++cell_index_y)
		{
			for(size_t cell_index_x = 0; cell_index_x < num_smallest_cells_in_axis_x; ++cell_index_x)
			{
				scm::math::vec3d current_pos = start_pos + scm::math::vec3d(smallest_cell_size.x * (double)cell_index_x, smallest_cell_size.y * (double)cell_index_y, smallest_cell_size.z * (double)cell_index_z);
				const view_cell_regular* current_cell = (const view_cell_regular*)input_grid->get_cell_at_position(current_pos, nullptr);

				// Collect visibility data.
				for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
				{
					boost::dynamic_bitset<> local_visibility = current_cell->get_bitset(model_index);
					local_visibility.resize(input_grid->get_num_nodes(model_index));

					outer_cell_visibility[model_index] = outer_cell_visibility[model_index] | local_visibility;
				}
			}
		}
	}

	// Finally copy gathered visibility data to boundary view cell.
	for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
	{
		for(node_t node_index = 0; node_index < input_grid->get_num_nodes(model_index); ++node_index)
		{
			output_grid->set_cell_visibility(output_cell_index, model_index, node_index, outer_cell_visibility[model_index][node_index]);
		}
	}
}

void pvs_bounds_extrapolator_from_outer_cells::
merge_visibilities_within_bounding_grid(const grid* input_grid, 
										grid_bounding* output_grid, 
										const size_t& output_cell_index, 
										const size_t& input_cell_index_one, 
										const size_t& input_cell_index_two) const
{
	std::vector<boost::dynamic_bitset<>> outer_cell_visibility;
	outer_cell_visibility.resize(input_grid->get_num_models());

	for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
	{
		outer_cell_visibility[model_index].resize(input_grid->get_num_nodes(model_index));
	}

	// Collect visibility data.
	for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
	{
		// Collect visibility data.
		for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
		{
			boost::dynamic_bitset<> local_visibility_one = ((const view_cell_regular*)output_grid->get_cell_at_index(input_cell_index_one))->get_bitset(model_index);
			boost::dynamic_bitset<> local_visibility_two = ((const view_cell_regular*)output_grid->get_cell_at_index(input_cell_index_two))->get_bitset(model_index);

			outer_cell_visibility[model_index] = local_visibility_one | local_visibility_two;
		}
	}

	// Finally copy gathered visibility data to boundary view cell.
	for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
	{
		for(node_t node_index = 0; node_index < input_grid->get_num_nodes(model_index); ++node_index)
		{
			output_grid->set_cell_visibility(output_cell_index, model_index, node_index, outer_cell_visibility[model_index][node_index]);
		}
	}
}

void pvs_bounds_extrapolator_from_outer_cells::
merge_visibilities_within_bounding_grid(const grid* input_grid, 
										grid_bounding* output_grid, 
										const size_t& output_cell_index, 
										const size_t& input_cell_index_one, 
										const size_t& input_cell_index_two,
										const size_t& input_cell_index_three) const
{
	std::vector<boost::dynamic_bitset<>> outer_cell_visibility;
	outer_cell_visibility.resize(input_grid->get_num_models());

	for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
	{
		outer_cell_visibility[model_index].resize(input_grid->get_num_nodes(model_index));
	}

	// Collect visibility data.
	for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
	{
		// Collect visibility data.
		for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
		{
			boost::dynamic_bitset<> local_visibility_one = ((const view_cell_regular*)output_grid->get_cell_at_index(input_cell_index_one))->get_bitset(model_index);
			boost::dynamic_bitset<> local_visibility_two = ((const view_cell_regular*)output_grid->get_cell_at_index(input_cell_index_two))->get_bitset(model_index);
			boost::dynamic_bitset<> local_visibility_three = ((const view_cell_regular*)output_grid->get_cell_at_index(input_cell_index_three))->get_bitset(model_index);

			outer_cell_visibility[model_index] = local_visibility_one | local_visibility_two | local_visibility_three;
		}
	}

	// Finally copy gathered visibility data to boundary view cell.
	for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
	{
		for(node_t node_index = 0; node_index < input_grid->get_num_nodes(model_index); ++node_index)
		{
			output_grid->set_cell_visibility(output_cell_index, model_index, node_index, outer_cell_visibility[model_index][node_index]);
		}
	}
}

}
}