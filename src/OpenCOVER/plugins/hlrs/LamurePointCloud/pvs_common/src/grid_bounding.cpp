// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/grid_bounding.h"
#include "lamure/pvs/view_cell_irregular.h"

#include <fstream>
#include <iostream>

namespace lamure
{
namespace pvs
{

grid_bounding::
grid_bounding() : grid_bounding(nullptr, std::vector<node_t>())
{
}

grid_bounding::
grid_bounding(const grid* core_grid, const std::vector<node_t>& ids)
{
	this->create_grid(core_grid);

	ids_.resize(ids.size());
	for(size_t index = 0; index < ids_.size(); ++index)
	{
		ids_[index] = ids[index];
	}
}

grid_bounding::
~grid_bounding()
{
}

std::string grid_bounding::
get_grid_type() const
{
	return get_grid_identifier();
}

std::string grid_bounding::
get_grid_identifier()
{
	return "bounding";
}

const view_cell* grid_bounding::
get_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const
{
	size_t general_index = 0;

	{
		std::lock_guard<std::mutex> lock(mutex_);

		view_cell* center_cell = cells_[13];

		int num_cells = 3;
		scm::math::vec3d cell_size = center_cell->get_size() * 0.5;
		scm::math::vec3d distance = position - position_center_;

		int index_x = (int)(distance.x / cell_size.x);
		int index_y = (int)(distance.y / cell_size.y);
		int index_z = (int)(distance.z / cell_size.z);

		// Normalize to value -1 or 1 if not 0.
		if(index_x != 0)
		{
			index_x = index_x / std::abs(index_x);
		}
		if(index_y != 0)
		{
			index_y = index_y / std::abs(index_y);
		}
		if(index_z != 0)
		{
			index_z = index_z / std::abs(index_z);
		}

		++index_x;
		++index_y;
		++index_z;

		general_index = (size_t)((num_cells * num_cells * index_z) + (num_cells * index_y) + index_x);
	}

	// Optional second return value: the index of the view cell.
	if(cell_index != nullptr)
	{
		(*cell_index) = general_index;
	}

	return get_cell_at_index(general_index);
}

void grid_bounding::
save_grid_to_file(const std::string& file_path) const
{
	this->save_bounding_grid(file_path, get_grid_identifier());
}


bool grid_bounding::
load_grid_from_file(const std::string& file_path)
{
	return this->load_bounding_grid(file_path, get_grid_identifier());
}

void grid_bounding::
create_grid(const grid* core_grid)
{
	for(size_t cell_index = 0; cell_index < cells_.size(); ++cell_index)
	{
		delete cells_[cell_index];
	}

	cells_.clear();
	cells_.resize(27);

	size_t cell_index = 0;
	scm::math::vec3d cell_size = core_grid->get_size();

	for(int z_dir = -1; z_dir <= 1; ++z_dir)
	{
		for(int y_dir = -1; y_dir <= 1; ++y_dir)
		{
			for(int x_dir = -1; x_dir <= 1; ++x_dir)
			{
				scm::math::vec3d position_center = core_grid->get_position_center() + cell_size * scm::math::vec3d(x_dir, y_dir, z_dir);
				cells_[cell_index] = new view_cell_irregular(cell_size, position_center);

				++cell_index;
			}
		}
	}

	size_ = cell_size * 3.0;
	position_center_ = scm::math::vec3d(core_grid->get_position_center());
}

void grid_bounding::
create_grid(const scm::math::vec3d& center_cell_size, const scm::math::vec3d& position_center)
{
	for(size_t cell_index = 0; cell_index < cells_.size(); ++cell_index)
	{
		delete cells_[cell_index];
	}

	cells_.clear();
	cells_.resize(27);

	size_t cell_index = 0;

	for(int z_dir = -1; z_dir <= 1; ++z_dir)
	{
		for(int y_dir = -1; y_dir <= 1; ++y_dir)
		{
			for(int x_dir = -1; x_dir <= 1; ++x_dir)
			{
				scm::math::vec3d local_position_center = position_center + center_cell_size * scm::math::vec3d(x_dir, y_dir, z_dir);
				cells_[cell_index] = new view_cell_irregular(center_cell_size, local_position_center);

				++cell_index;
			}
		}
	}

	size_ = center_cell_size * 3.0;
	position_center_ = position_center;
}

void grid_bounding::
save_bounding_grid(const std::string& file_path, const std::string& grid_type) const
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_out;
	file_out.open(file_path, std::ios::out);

	if(!file_out.is_open())
	{
		throw std::invalid_argument("invalid file path: " + file_path);
	}

	// Grid file type.
	file_out << grid_type << std::endl;

	// Center cell size and position.
	file_out << cells_[13]->get_size().x << " " << cells_[13]->get_size().y << " " << cells_[13]->get_size().z << std::endl;
	file_out << position_center_.x << " " << position_center_.y << " " << position_center_.z << std::endl;

	// Save number of models, so we can later simply read the node numbers.
	file_out << ids_.size() << std::endl;

	// Save the number of node ids of each model.
	for(size_t model_index = 0; model_index < ids_.size(); ++model_index)
	{
		file_out << ids_[model_index] << " ";
	}

	file_out.close();
}

bool grid_bounding::
load_bounding_grid(const std::string& file_path, const std::string& grid_type)
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_in;
	file_in.open(file_path, std::ios::in);

	if(!file_in.is_open())
	{
		std::cout << "Not able to open file: " << file_path << std::endl;
		return false;
	}

	// Start reading the header info which is used to recreate the grid.
	std::string input_grid_type;
	file_in >> input_grid_type;
	if(input_grid_type != grid_type)
	{
		std::cout << "Wrong grid type: '" << input_grid_type <<  "'' / '" << grid_type << "'" << std::endl;
		return false;
	}

	double cell_size_x, cell_size_y, cell_size_z;
	file_in >> cell_size_x >> cell_size_y >> cell_size_z;

	double pos_x, pos_y, pos_z;
	file_in >> pos_x >> pos_y >> pos_z;

	scm::math::vec3d center_cell_size(cell_size_x, cell_size_y, cell_size_z);
	scm::math::vec3d position_center(pos_x, pos_y, pos_z);

	this->create_grid(center_cell_size, position_center_);

	// Read the number of models.
	size_t num_models = 0;
	file_in >> num_models;
	ids_.resize(num_models);

	// Read the number of nodes per model.
	for(size_t model_index = 0; model_index < ids_.size(); ++model_index)
	{
		node_t num_nodes = 0;
		file_in >> num_nodes;
		ids_[model_index] = num_nodes;
	}

	file_in.close();
	return true;
}

}
}
