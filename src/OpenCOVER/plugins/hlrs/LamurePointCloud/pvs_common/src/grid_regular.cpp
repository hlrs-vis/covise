// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/grid_regular.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <climits>
#include <iostream>

namespace lamure
{
namespace pvs
{

grid_regular::
grid_regular() : grid_regular(1, 1.0, scm::math::vec3d(0.0, 0.0, 0.0), std::vector<node_t>())
{
}

grid_regular::
grid_regular(const size_t& number_cells, const double& bounds_size, const scm::math::vec3d& position_center, const std::vector<node_t>& ids)
{
	create_grid(number_cells, bounds_size / (double)number_cells, position_center);
	
	ids_.resize(ids.size());
	for(size_t index = 0; index < ids_.size(); ++index)
	{
		ids_[index] = ids[index];
	}
}

grid_regular::
~grid_regular()
{
	for(size_t cell_index = 0; cell_index < cells_.size(); ++cell_index)
	{
		delete cells_[cell_index];
	}

	cells_.clear();
}

std::string grid_regular::
get_grid_type() const
{
	return get_grid_identifier();
}

std::string grid_regular::
get_grid_identifier()
{
	return "regular";
}

size_t grid_regular::
get_cell_count() const
{
	return cells_.size();
}

scm::math::vec3d grid_regular::
get_size() const
{
	return size_;
}

scm::math::vec3d grid_regular::
get_position_center() const
{
	return position_center_;
}

const view_cell* grid_regular::
get_cell_at_index(const size_t& index) const
{
	std::lock_guard<std::mutex> lock(mutex_);

	return cells_[index];
}

const view_cell* grid_regular::
get_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const
{
	return calculate_cell_at_position(position, cell_index);
}

view_cell* grid_regular::
calculate_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const
{
	size_t general_index = 0;

	{
		std::lock_guard<std::mutex> lock(mutex_);

		size_t num_cells = std::pow(cells_.size(), 1.0f/3.0f);
		double half_size = cell_size_ * (double)num_cells * 0.5f;
		scm::math::vec3d distance = position - (position_center_ - half_size);

		size_t index_x = (size_t)(distance.x / cell_size_);
		size_t index_y = (size_t)(distance.y / cell_size_);
		size_t index_z = (size_t)(distance.z / cell_size_);

		// Check calculated index so we know if the position is inside the grid at all.
		if(index_x < 0 || index_x >= num_cells ||
			index_y < 0 || index_y >= num_cells ||
			index_z < 0 || index_z >= num_cells)
		{
			return nullptr;
		}

		general_index = (num_cells * num_cells * index_z) + (num_cells * index_y) + index_x;
	}

	// Optional second return value: the index of the view cell.
	if(cell_index != nullptr)
	{
		(*cell_index) = general_index;
	}

	return cells_[general_index];
}

void grid_regular::
set_cell_visibility(const size_t& cell_index, const model_t& model_id, const node_t& node_id, const bool& visibility)
{
	// If this function is locked, high performance loss in the preprocessing will occur.
	//std::lock_guard<std::mutex> lock(mutex_);
	
	view_cell* current_visibility_cell = cells_[cell_index];
	current_visibility_cell->set_visibility(model_id, node_id, visibility);
}

void grid_regular::
set_cell_visibility(const scm::math::vec3d& position, const model_t& model_id, const node_t& node_id, const bool& visibility)
{
	view_cell* current_visibility_cell = this->calculate_cell_at_position(position, nullptr);

	if(current_visibility_cell != nullptr)
	{
		current_visibility_cell->set_visibility(model_id, node_id, visibility);
	}
}

void grid_regular::
save_grid_to_file(const std::string& file_path) const
{
	save_regular_grid(file_path, get_grid_identifier());
}

void grid_regular::
save_regular_grid(const std::string& file_path, const std::string& grid_type) const
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

	// Number of grid cells per dimension.
	file_out << std::pow(cells_.size(), 1.0f/3.0f) << std::endl;

	// Grid size and position
	file_out << cell_size_ << std::endl;
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

void grid_regular::
save_visibility_to_file(const std::string& file_path) const
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_out;
	file_out.open(file_path, std::ios::out | std::ios::binary);

	if(!file_out.is_open())
	{
		throw std::invalid_argument("invalid file path: " + file_path);
	}

	std::vector<std::string> compressed_data_blocks;

	// Iterate over view cells.
	for(size_t cell_index = 0; cell_index < cells_.size(); ++cell_index)
	{
		std::string current_cell_data = "";

		// Iterate over models in the scene.
		for(lamure::model_t model_id = 0; model_id < ids_.size(); ++model_id)
		{
			node_t num_nodes = ids_.at(model_id);
			char current_byte = 0x00;

			size_t line_length = num_nodes / CHAR_BIT + (num_nodes % CHAR_BIT == 0 ? 0 : 1);
			size_t character_counter = 0;
			std::string current_line_data(line_length, 0x00);

			// Iterate over nodes in the model.
			for(lamure::node_t node_id = 0; node_id < num_nodes; ++node_id)
			{
				if(cells_[cell_index]->get_visibility(model_id, node_id))
				{
					current_byte |= 1 << (node_id % CHAR_BIT);
				}

				// Flush character if either 8 bits are written or if the node id is the last one.
				if((node_id + 1) % CHAR_BIT == 0 || node_id == (num_nodes - 1))
				{
					current_line_data[character_counter] = current_byte;
					character_counter++;

					current_byte = 0x00;
				}
			}

			current_cell_data = current_cell_data + current_line_data;
		}

		file_out.write(current_cell_data.c_str(), current_cell_data.length());
	}

	file_out.close();
}

bool grid_regular::
load_grid_from_file(const std::string& file_path)
{
	return load_regular_grid(file_path, get_grid_identifier());
}

bool grid_regular::
load_regular_grid(const std::string& file_path, const std::string& grid_type)
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

	size_t num_cells;
	file_in >> num_cells;

	double cell_size;
	file_in >> cell_size;

	double pos_x, pos_y, pos_z;
	file_in >> pos_x >> pos_y >> pos_z;

	position_center_ = scm::math::vec3d(pos_x, pos_y, pos_z);
	create_grid(num_cells, cell_size, position_center_);

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


bool grid_regular::
load_visibility_from_file(const std::string& file_path)
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_in;
	file_in.open(file_path, std::ios::in | std::ios::binary);

	if(!file_in.is_open())
	{
		return false;
	}

	for(size_t cell_index = 0; cell_index < cells_.size(); ++cell_index)
	{
		view_cell* current_cell = cells_[cell_index];

		// One line per model.
		for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
		{
			node_t num_nodes = ids_.at(model_index);
			size_t line_length = num_nodes / CHAR_BIT + (num_nodes % CHAR_BIT == 0 ? 0 : 1);

      std::vector<char> current_line_data(line_length);

			file_in.read(&current_line_data[0], line_length);

			// Used to avoid continuing resize within visibility data.
			current_cell->set_visibility(model_index, num_nodes - 1, false);

			for(node_t character_index = 0; character_index < line_length; ++character_index)
			{
				char current_byte = current_line_data[character_index];
				
				for(unsigned short bit_index = 0; bit_index < CHAR_BIT; ++bit_index)
				{
					bool visible = ((current_byte >> bit_index) & 1) == 0x01;
					current_cell->set_visibility(model_index, (character_index * CHAR_BIT) + bit_index, visible);
				}
			}
		}
	}

	file_in.close();
	return true;
}

void grid_regular::
clear_cell_visibility(const size_t& cell_index)
{
	if(this->get_num_models() <= cell_index)
	{
		return;
	}

	std::lock_guard<std::mutex> lock(mutex_);

	cells_[cell_index]->clear_visibility_data();
}

bool grid_regular::
load_cell_visibility_from_file(const std::string& file_path, const size_t& cell_index)
{
	std::lock_guard<std::mutex> lock(mutex_);

	view_cell* current_cell = cells_[cell_index];

	// First check if visibility data is already loaded.
	if(current_cell->contains_visibility_data())
	{
		return true;
	}

	// If no visibility data exists, open the file and load them.
	std::fstream file_in;
	file_in.open(file_path, std::ios::in | std::ios::binary);

	if(!file_in.is_open())
	{
		return false;
	}

	// Calculate the number of bytes of a view cell (note: every view cell requires same storage).
	node_t single_cell_bytes = 0;

	for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
	{
		node_t num_nodes = ids_.at(model_index);

		// If the number of node IDs is not dividable by 8 there is one additional character.
		node_t addition = 0;
		if(num_nodes % CHAR_BIT != 0)
		{
			addition = 1;
		}

		node_t line_size = (num_nodes / CHAR_BIT) + addition;
		single_cell_bytes += line_size;
	}

	// Move to proper start point within file.
	file_in.seekg(cell_index * single_cell_bytes);

	// Read the visibility data.
	for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
	{
		node_t num_nodes = ids_.at(model_index);
		size_t line_length = num_nodes / CHAR_BIT + (num_nodes % CHAR_BIT == 0 ? 0 : 1);
    std::vector<char> current_line_data(line_length);

		file_in.read(&current_line_data[0], line_length);

		// Used to avoid continuing resize within visibility data.
		current_cell->set_visibility(model_index, num_nodes - 1, false);

		for(node_t character_index = 0; character_index < line_length; ++character_index)
		{
			char current_byte = current_line_data[character_index];
			
			for(unsigned short bit_index = 0; bit_index < CHAR_BIT; ++bit_index)
			{
				bool visible = ((current_byte >> bit_index) & 1) == 0x01;
				current_cell->set_visibility(model_index, (character_index * CHAR_BIT) + bit_index, visible);
			}
		}
	}

	file_in.close();

	return true;
}

void grid_regular::
create_grid(const size_t& num_cells, const double& cell_size, const scm::math::vec3d& position_center)
{
	for(size_t cell_index = 0; cell_index < cells_.size(); ++cell_index)
	{
		delete cells_[cell_index];
	}
	
	cells_.clear();

	double half_size = (cell_size * (double)num_cells) * 0.5;		// position of grid is at grid center, so cells have a offset
	double cell_offset = cell_size * 0.5f;							// position of cell is at cell center

	for(size_t index_z = 0; index_z < num_cells; ++index_z)
	{
		for(size_t index_y = 0; index_y < num_cells; ++index_y)
		{
			for(size_t index_x = 0; index_x < num_cells; ++index_x)
			{
				scm::math::vec3d pos = position_center + (scm::math::vec3d(index_x , index_y, index_z) * cell_size) - half_size + cell_offset;
				cells_.push_back(new view_cell_regular(cell_size, pos));
			}
		}
	}

	cell_size_ = cell_size;
	size_ = scm::math::vec3d((double)num_cells * cell_size, (double)num_cells * cell_size, (double)num_cells * cell_size);
	position_center_ = scm::math::vec3d(position_center);
}

model_t grid_regular::
get_num_models() const
{
	return ids_.size();
}

node_t grid_regular::
get_num_nodes(const model_t& model_id) const
{
	return ids_[model_id];
}

}
}
