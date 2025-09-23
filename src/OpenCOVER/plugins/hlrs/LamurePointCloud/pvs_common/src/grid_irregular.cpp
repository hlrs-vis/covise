// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/grid_irregular.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <climits>
#include <iostream>

namespace lamure
{
namespace pvs
{

grid_irregular::
grid_irregular() : grid_irregular(1, 1, 1, 1.0, scm::math::vec3d(0.0, 0.0, 0.0), std::vector<node_t>())
{
}

grid_irregular::
grid_irregular(const size_t& number_cells_x, const size_t& number_cells_y, const size_t& number_cells_z, const double& bounds_size, const scm::math::vec3d& position_center, const std::vector<node_t>& ids)
{
	size_t longest_axis_cell_count = std::max(number_cells_x, std::max(number_cells_y, number_cells_z));
	double cell_size = bounds_size / (double)longest_axis_cell_count;

	create_grid(number_cells_x, number_cells_y, number_cells_z, cell_size, position_center);
	
	ids_.resize(ids.size());
	for(size_t index = 0; index < ids_.size(); ++index)
	{
		ids_[index] = ids[index];
	}
}

grid_irregular::
~grid_irregular()
{
	original_cells_.clear();
	managing_cells_.clear();
	cells_by_indices_.clear();
}

std::string grid_irregular::
get_grid_type() const
{
	return get_grid_identifier();
}

std::string grid_irregular::
get_grid_identifier()
{
	return "irregular";
}

size_t grid_irregular::
get_cell_count() const
{
	return cells_by_indices_.size();
}

scm::math::vec3d grid_irregular::
get_size() const
{
	return size_;
}

scm::math::vec3d grid_irregular::
get_position_center() const
{
	return position_center_;
}

const view_cell* grid_irregular::
get_cell_at_index(const size_t& index) const
{
	std::lock_guard<std::mutex> lock(mutex_);

	return cells_by_indices_[index];
}

const view_cell* grid_irregular::
get_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const
{
	return this->calculate_cell_at_position(position, cell_index);
}

view_cell* grid_irregular::
calculate_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const
{
	size_t general_index = 0;

	{
		std::lock_guard<std::mutex> lock(mutex_);

		size_t original_index = 0;
		const view_cell* original_cell = this->get_original_cell_at_position(position, &original_index);

		// This means the position is outside of the grid.
		if(original_cell == nullptr)
		{
			return nullptr;
		}

		if(cells_active_states_[original_index])
		{
			// Cell is not joined, so the cell must be among the indices for fast access.
			for(size_t indexed_cell_index = 0; indexed_cell_index < cells_by_indices_.size(); ++indexed_cell_index)
			{
				if(original_cell == cells_by_indices_[indexed_cell_index])
				{
					general_index = indexed_cell_index;
					break;
				}
			}
		}
		else
		{
			// Cell is joined, so it must be among the managing view cells.
			size_t managed_index = original_index_to_cell_mapping_.at(original_index);
			const view_cell* managed_cell = &managing_cells_[managed_index];

			for(size_t indexed_cell_index = 0; indexed_cell_index < cells_by_indices_.size(); ++indexed_cell_index)
			{
				if(managed_cell == cells_by_indices_[indexed_cell_index])
				{
					general_index = indexed_cell_index;
					break;
				}
			}
		}
	}

	// Optional second return value: the index of the view cell.
	if(cell_index != nullptr)
	{
		(*cell_index) = general_index;
	}

	return cells_by_indices_[general_index];
}

void grid_irregular::
set_cell_visibility(const size_t& cell_index, const model_t& model_id, const node_t& node_id, const bool& visibility)
{
	view_cell* current_visibility_cell = cells_by_indices_[cell_index];
	current_visibility_cell->set_visibility(model_id, node_id, visibility);
}

void grid_irregular::
set_cell_visibility(const scm::math::vec3d& position, const model_t& model_id, const node_t& node_id, const bool& visibility)
{
	view_cell* current_visibility_cell = calculate_cell_at_position(position, nullptr);
	
	if(current_visibility_cell != nullptr)
	{
		current_visibility_cell->set_visibility(model_id, node_id, visibility);
	}
}

void grid_irregular::
save_grid_to_file(const std::string& file_path) const
{
	save_irregular_grid(file_path, get_grid_identifier());
}

void grid_irregular::
save_irregular_grid(const std::string& file_path, const std::string& grid_type) const
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
	file_out << number_cells_x_ << " " << number_cells_y_ << " " << number_cells_z_ << std::endl;

	// Grid size and position
	file_out << cell_size_ << std::endl;
	file_out << position_center_.x << " " << position_center_.y << " " << position_center_.z << std::endl;

	// Number of mapping entires.
	file_out << original_index_to_cell_mapping_.size() << std::endl;

	// Joined view cell IDs.
	for(std::map<size_t, size_t>::const_iterator iter = original_index_to_cell_mapping_.begin(); iter != original_index_to_cell_mapping_.end(); ++iter)
	{
		file_out << iter->first << " " << iter->second << std::endl;
	}

	// Save number of models, so we can later simply read the node numbers.
	file_out << ids_.size() << std::endl;

	// Save the number of node ids of each model.
	for(size_t model_index = 0; model_index < ids_.size(); ++model_index)
	{
		file_out << ids_[model_index] << " ";
	}

	file_out.close();
}

void grid_irregular::
save_visibility_to_file(const std::string& file_path) const
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_out;
	file_out.open(file_path, std::ios::out | std::ios::binary);

	if(!file_out.is_open())
	{
		throw std::invalid_argument("invalid file path: " + file_path);
	}

	// Iterate over view cells.
	for(size_t cell_index = 0; cell_index < cells_by_indices_.size(); ++cell_index)
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
				if(cells_by_indices_[cell_index]->get_visibility(model_id, node_id))
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

bool grid_irregular::
load_grid_from_file(const std::string& file_path)
{
	return load_irregular_grid(file_path, get_grid_identifier());
}

bool grid_irregular::
load_irregular_grid(const std::string& file_path, const std::string& grid_type)
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

	size_t num_cells_x, num_cells_y, num_cells_z;
	file_in >> num_cells_x >> num_cells_y >> num_cells_z;

	double cell_size;
	file_in >> cell_size;

	double pos_x, pos_y, pos_z;
	file_in >> pos_x >> pos_y >> pos_z;

	position_center_ = scm::math::vec3d(pos_x, pos_y, pos_z);
	create_grid(num_cells_x, num_cells_y, num_cells_z, cell_size, position_center_);

	// Number of mapping entires.
	size_t num_mappings;
	file_in >> num_mappings;

	// Joined view cell IDs.
	original_index_to_cell_mapping_.clear();
	size_t num_managing_cells = 0;

	for(size_t mapping_index = 0; mapping_index < num_mappings; ++mapping_index)
	{
		size_t original_index, managing_index;
		file_in >> original_index >> managing_index;

		cells_active_states_[original_index] = false;
		original_index_to_cell_mapping_[original_index] = managing_index;

		// Track highest managing cell index to get total number of managing cells.
		if(managing_index + 1 > num_managing_cells)
		{
			num_managing_cells = managing_index + 1;
		}
	}

	// Create managing cells.
	managing_cells_.resize(num_managing_cells);

	for(std::map<size_t, size_t>::const_iterator iter = original_index_to_cell_mapping_.begin(); iter != original_index_to_cell_mapping_.end(); ++iter)
	{
		managing_cells_[iter->second].add_cell(&original_cells_[iter->first]);
	}

	// Cell layout was changed, reindexing required.
	this->compute_index_access();

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


bool grid_irregular::
load_visibility_from_file(const std::string& file_path)
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_in;
	file_in.open(file_path, std::ios::in | std::ios::binary);

	if(!file_in.is_open())
	{
		return false;
	}

	for(size_t cell_index = 0; cell_index < cells_by_indices_.size(); ++cell_index)
	{
		view_cell* current_cell = cells_by_indices_[cell_index];

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

void grid_irregular::
clear_cell_visibility(const size_t& cell_index)
{
	std::lock_guard<std::mutex> lock(mutex_);

	cells_by_indices_[cell_index]->clear_visibility_data();
}

bool grid_irregular::
load_cell_visibility_from_file(const std::string& file_path, const size_t& cell_index)
{
	std::lock_guard<std::mutex> lock(mutex_);

	view_cell* current_cell = cells_by_indices_[cell_index];

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

void grid_irregular::
create_grid(const size_t& number_cells_x, const size_t& number_cells_y, const size_t& number_cells_z, const double& cell_size, const scm::math::vec3d& position_center)
{
	original_cells_.clear();
	managing_cells_.clear();
	cells_active_states_.clear();
	cells_by_indices_.clear();
	original_index_to_cell_mapping_.clear();

	// position of grid is at grid center, so cells have a offset
	double half_size_x = (cell_size * (double)number_cells_x) * 0.5;
	double half_size_y = (cell_size * (double)number_cells_y) * 0.5;
	double half_size_z = (cell_size * (double)number_cells_z) * 0.5;
	scm::math::vec3d half_size(half_size_x, half_size_y, half_size_z);

	// position of cell is at cell center	
	double cell_offset = cell_size * 0.5f;

	for(size_t index_z = 0; index_z < number_cells_z; ++index_z)
	{
		for(size_t index_y = 0; index_y < number_cells_y; ++index_y)
		{
			for(size_t index_x = 0; index_x < number_cells_x; ++index_x)
			{
				scm::math::vec3d pos = position_center + (scm::math::vec3d(index_x , index_y, index_z) * cell_size) - half_size + cell_offset;
				original_cells_.push_back(view_cell_regular(cell_size, pos));
				cells_active_states_.push_back(true);
			}
		}
	}

	for(size_t original_cell_index = 0; original_cell_index < original_cells_.size(); ++original_cell_index)
	{
		cells_by_indices_.push_back(&original_cells_[original_cell_index]);
	}

	number_cells_x_ = number_cells_x;
	number_cells_y_ = number_cells_y;
	number_cells_z_ = number_cells_z;

	cell_size_ = cell_size;
	size_ = scm::math::vec3d((double)number_cells_x_ * cell_size, (double)number_cells_y_ * cell_size, (double)number_cells_z_ * cell_size);
	position_center_ = scm::math::vec3d(position_center);
}

model_t grid_irregular::
get_num_models() const
{
	return ids_.size();
}

node_t grid_irregular::
get_num_nodes(const model_t& model_id) const
{
	return ids_[model_id];
}

void grid_irregular::
compute_index_access()
{
	cells_by_indices_.clear();

	for(size_t original_cell_index = 0; original_cell_index < original_cells_.size(); ++original_cell_index)
	{
		if(cells_active_states_[original_cell_index])
		{
			cells_by_indices_.push_back(&original_cells_[original_cell_index]);
		}
	}

	for(size_t managing_cell_index = 0; managing_cell_index < managing_cells_.size(); ++managing_cell_index)
	{
		cells_by_indices_.push_back(&managing_cells_[managing_cell_index]);
	}
}

size_t grid_irregular::
get_original_cell_count() const
{
	return original_cells_.size();
}

bool grid_irregular::
join_cells(const size_t& index_one, const size_t& index_two, const float& error, const float& equality_threshold)
{
	bool result = false;

	if(index_one >= cells_by_indices_.size() ||
		index_two >= cells_by_indices_.size())
	{
		return false;
	}

	view_cell* view_cell_one = cells_by_indices_[index_one];
	view_cell* view_cell_two = cells_by_indices_[index_two];

	// Type of cells must be known (original or managing).
	bool view_cell_one_is_original = this->is_cell_at_index_original(index_one);
	bool view_cell_two_is_original = this->is_cell_at_index_original(index_two);

	// Join cells.
	if(view_cell_one_is_original && view_cell_two_is_original)
	{
		if((1.0f - error) >= equality_threshold)
		{
			// Case 1: both cells are original cells.
			size_t original_index_one = this->get_original_index_of_cell(view_cell_one);
			size_t original_index_two = this->get_original_index_of_cell(view_cell_two);

			view_cell_regular_managing new_managing_cell;
			new_managing_cell.add_cell(view_cell_one);
			new_managing_cell.add_cell(view_cell_two);
		
			// Copy visibility data.
			view_cell_regular* view_cell_regular_one = (view_cell_regular*)view_cell_one;
			view_cell_regular* view_cell_regular_two = (view_cell_regular*)view_cell_two;

			for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
			{
				boost::dynamic_bitset<> bitset_one = boost::dynamic_bitset<>(view_cell_regular_one->get_bitset(model_index));
				bitset_one.resize(this->get_num_nodes(model_index));
				boost::dynamic_bitset<> bitset_two = boost::dynamic_bitset<>(view_cell_regular_two->get_bitset(model_index));
				bitset_two.resize(this->get_num_nodes(model_index));

				boost::dynamic_bitset<> combined_visibility = bitset_one | bitset_two;
				new_managing_cell.set_bitset(model_index, combined_visibility);
			}

			new_managing_cell.set_error(error);
			managing_cells_.push_back(new_managing_cell);

			// Original view cells don't need to manage visibility data anymore.
			view_cell_one->clear_visibility_data();
			view_cell_two->clear_visibility_data();

			cells_active_states_[original_index_one] = false;
			cells_active_states_[original_index_two] = false;

			// Save mapping. Mapped to currently last cell in managing view cells.
			original_index_to_cell_mapping_[original_index_one] = managing_cells_.size() - 1;
			original_index_to_cell_mapping_[original_index_two] = managing_cells_.size() - 1;

			result = true;
		}
	}
	else if(view_cell_one_is_original)
	{
		// Case 2a: first index is original cell.
		size_t original_index_one = this->get_original_index_of_cell(view_cell_one);
		size_t managing_index_two = this->get_managing_index_of_cell(view_cell_two);

		view_cell_regular_managing* cell_managing = &managing_cells_[managing_index_two];

		if((1.0 - error) - cell_managing->get_error() >= equality_threshold)
		{
			// Copy visibility data.
			view_cell_regular* view_cell_regular_one = (view_cell_regular*)view_cell_one;
			view_cell_regular* view_cell_regular_two = (view_cell_regular*)view_cell_two;

			for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
			{
				boost::dynamic_bitset<> bitset_one = boost::dynamic_bitset<>(view_cell_regular_one->get_bitset(model_index));
				bitset_one.resize(this->get_num_nodes(model_index));
				boost::dynamic_bitset<> bitset_two = boost::dynamic_bitset<>(view_cell_regular_two->get_bitset(model_index));
				bitset_two.resize(this->get_num_nodes(model_index));

				boost::dynamic_bitset<> combined_visibility = bitset_one | bitset_two;
				view_cell_regular_two->set_bitset(model_index, combined_visibility);
			}

			// Original view cells don't need to manage visibility data anymore.
			view_cell_one->clear_visibility_data();
			cells_active_states_[original_index_one] = false;

			// Save mapping.
			original_index_to_cell_mapping_[original_index_one] = managing_index_two;

			cell_managing->add_cell(view_cell_one);
			cell_managing->set_error(error + cell_managing->get_error());

			result = true;
		}
	}
	else if(view_cell_two_is_original)
	{
		// Case 2b: second index is original cell.
		size_t managing_index_one = this->get_managing_index_of_cell(view_cell_one);
		size_t original_index_two = this->get_original_index_of_cell(view_cell_two);

		view_cell_regular_managing* cell_managing = &managing_cells_[managing_index_one];

		if((1.0 - error) - cell_managing->get_error() >= equality_threshold)
		{
			// Copy visibility data.
			view_cell_regular* view_cell_regular_one = (view_cell_regular*)view_cell_one;
			view_cell_regular* view_cell_regular_two = (view_cell_regular*)view_cell_two;

			for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
			{
				boost::dynamic_bitset<> bitset_one = boost::dynamic_bitset<>(view_cell_regular_one->get_bitset(model_index));
				bitset_one.resize(this->get_num_nodes(model_index));
				boost::dynamic_bitset<> bitset_two = boost::dynamic_bitset<>(view_cell_regular_two->get_bitset(model_index));
				bitset_two.resize(this->get_num_nodes(model_index));

				boost::dynamic_bitset<> combined_visibility = bitset_one | bitset_two;
				view_cell_regular_one->set_bitset(model_index, combined_visibility);
			}

			// Original view cells don't need to manage visibility data anymore.
			view_cell_two->clear_visibility_data();
			cells_active_states_[original_index_two] = false;

			// Save mapping.
			original_index_to_cell_mapping_[original_index_two] = managing_index_one;

			cell_managing->add_cell(view_cell_two);
			cell_managing->set_error(error + cell_managing->get_error());

			result = true;
		}
	}
	else
	{
		// Case 3: both are managing cells.
		size_t managing_index_one = this->get_managing_index_of_cell(view_cell_one);
		size_t managing_index_two = this->get_managing_index_of_cell(view_cell_two);

		float combined_error = managing_cells_[managing_index_one].get_error() + managing_cells_[managing_index_two].get_error();

		if((1.0 - error) - combined_error >= equality_threshold)
		{
			// Copy visibility data.
			view_cell_regular* view_cell_regular_one = (view_cell_regular*)view_cell_one;
			view_cell_regular* view_cell_regular_two = (view_cell_regular*)view_cell_two;

			for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
			{
				boost::dynamic_bitset<> bitset_one = boost::dynamic_bitset<>(view_cell_regular_one->get_bitset(model_index));
				bitset_one.resize(this->get_num_nodes(model_index));
				boost::dynamic_bitset<> bitset_two = boost::dynamic_bitset<>(view_cell_regular_two->get_bitset(model_index));
				bitset_two.resize(this->get_num_nodes(model_index));

				boost::dynamic_bitset<> combined_visibility = bitset_one | bitset_two;
				view_cell_regular_one->set_bitset(model_index, combined_visibility);
			}

			// Rewrite mapping and move managed view cells.
			for(std::map<size_t, size_t>::const_iterator iter = original_index_to_cell_mapping_.begin(); iter != original_index_to_cell_mapping_.end(); ++iter)
			{
				if(iter->second == managing_index_two)
				{
					original_index_to_cell_mapping_[iter->first] = managing_index_one;
					managing_cells_[managing_index_one].add_cell(&original_cells_[iter->first]);
				}
			}

			managing_cells_[managing_index_one].set_error(combined_error + error);

			// Remove the now ununsed managing view cell.
			managing_cells_.erase(managing_cells_.begin() + managing_index_two);

			// Now that one managing view cell was removed, every mapping with index higher than the removed element must be lowered by one.
			for(std::map<size_t, size_t>::const_iterator iter = original_index_to_cell_mapping_.begin(); iter != original_index_to_cell_mapping_.end(); ++iter)
			{
				if(iter->second > managing_index_two)
				{
					original_index_to_cell_mapping_[iter->first] = iter->second - 1;
				}
			}

			result = true;
		}
	}

	// Keep indices for access up to date.
	this->compute_index_access();

	return result;
}

const view_cell* grid_irregular::
get_original_cell_at_index(const size_t& index) const
{
	return &original_cells_[index];
}

const view_cell* grid_irregular::
get_original_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const
{
	size_t general_index = 0;

	// position of grid is at grid center, so cells have a offset
	double half_size_x = (cell_size_ * (double)number_cells_x_) * 0.5;
	double half_size_y = (cell_size_ * (double)number_cells_y_) * 0.5;
	double half_size_z = (cell_size_ * (double)number_cells_z_) * 0.5;
	scm::math::vec3d half_size(half_size_x, half_size_y, half_size_z);

	scm::math::vec3d distance = position - (position_center_ - half_size);

	size_t index_x = (size_t)(distance.x / cell_size_);
	size_t index_y = (size_t)(distance.y / cell_size_);
	size_t index_z = (size_t)(distance.z / cell_size_);

	// Check calculated index so we know if the position is inside the grid at all.
	if(index_x < 0 || index_x >= number_cells_x_ ||
		index_y < 0 || index_y >= number_cells_y_ ||
		index_z < 0 || index_z >= number_cells_z_)
	{
		return nullptr;
	}

	general_index = (number_cells_y_ * number_cells_x_ * index_z) + (number_cells_x_ * index_y) + index_x;

	// Optional second return value: the index of the view cell.
	if(cell_index != nullptr)
	{
		(*cell_index) = general_index;
	}

	return get_original_cell_at_index(general_index);
}

bool grid_irregular::
is_cell_at_index_original(const size_t& index) const
{
	return cells_by_indices_[index]->get_cell_type() == view_cell_regular::get_cell_identifier();
}

size_t grid_irregular::
get_original_index_of_cell(const view_cell* cell) const
{
	for(size_t original_cell_index = 0; original_cell_index < original_cells_.size(); ++original_cell_index)
	{
		if(cell == &original_cells_[original_cell_index])
		{
			return original_cell_index;
		}
	}

	return original_cells_.size();
}

size_t grid_irregular::
get_managing_index_of_cell(const view_cell* cell) const
{
	for(size_t managing_cell_index = 0; managing_cell_index < managing_cells_.size(); ++managing_cell_index)
	{
		if(cell == &managing_cells_[managing_cell_index])
		{
			return managing_cell_index;
		}
	}

	return managing_cells_.size();
}

}
}
