// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <fstream>
#include <climits>
#include <deque>
#include <iostream>

#include "lamure/bounding_box.h"
#include "lamure/pvs/grid_octree.h"
#include "lamure/pvs/view_cell_regular.h"

namespace lamure
{
namespace pvs
{

grid_octree::
grid_octree() : grid_octree(1, 1.0, scm::math::vec3d(0.0, 0.0, 0.0), std::vector<node_t>())
{
}

grid_octree::
grid_octree(const size_t& octree_depth, const double& size, const scm::math::vec3d& position_center, const std::vector<node_t>& ids)
{
	root_node_ = new grid_octree_node(size, position_center);
	
	size_t depth = octree_depth;
	create_grid(root_node_, depth);
	
	compute_index_access();

	ids_.resize(ids.size());
	for(size_t index = 0; index < ids_.size(); ++index)
	{
		ids_[index] = ids[index];
	}
}

grid_octree::
~grid_octree()
{
	if(root_node_ != nullptr)
	{
		delete root_node_;
	}
}

std::string grid_octree::
get_grid_type() const
{
	return get_grid_identifier();
}

std::string grid_octree::
get_grid_identifier()
{
	return "octree";
}

void grid_octree::
create_grid(grid_octree_node* node, size_t depth)
{
	depth--;

	if(depth > 0)
	{
		node->split();

		for(size_t child_index = 0; child_index < 8; ++child_index)
		{
			create_grid(node->get_child_at_index(child_index), depth);
		}
	}
}

void grid_octree::
compute_index_access()
{
	size_t num_cells = cell_count_recursive(root_node_);
	
	cells_by_indices_.clear();
	cells_by_indices_.resize(num_cells);

	for(size_t cell_index = 0; cell_index < num_cells; ++cell_index)
	{
		view_cell* cell_at_index = find_cell_by_index_recursive(root_node_, cell_index, 0);
		cells_by_indices_[cell_index] = cell_at_index;
	}
}

size_t grid_octree::
get_cell_count() const
{
	std::lock_guard<std::mutex> lock(mutex_);

	return cells_by_indices_.size();
}

scm::math::vec3d grid_octree::
get_size() const
{
	return root_node_->get_size();
}

scm::math::vec3d grid_octree::
get_position_center() const
{
	return root_node_->get_position_center();
}

size_t grid_octree::
cell_count_recursive(const grid_octree_node* node) const
{
	// No instance = no grid cell.
	if(node == nullptr)
	{
		return 0;
	}

	if(node->has_children())
	{
		size_t result = 0;

		for(size_t child_index = 0; child_index < 8; ++child_index)
		{
			result += cell_count_recursive(node->get_child_at_index_const(child_index));
		}

		return result;
	}
	else
	{
		return 1;
	}
}

const view_cell* grid_octree::
get_cell_at_index(const size_t& index) const
{
	std::lock_guard<std::mutex> lock(mutex_);

	return cells_by_indices_[index];
}

grid_octree_node* grid_octree::
find_cell_by_index_recursive(const grid_octree_node* node, const size_t& index, size_t base_value)
{
	// Ugly, but does the job. 
	// Source: http://stackoverflow.com/questions/123758/how-do-i-remove-code-duplication-between-similar-const-and-non-const-member-func
	return const_cast<grid_octree_node*>(static_cast<const grid_octree&>(*this).find_cell_by_index_recursive_const(node, index, base_value));
}

const grid_octree_node* grid_octree::
find_cell_by_index_recursive_const(const grid_octree_node* node, const size_t& index, size_t base_value) const
{
	size_t num_cells = cell_count_recursive(node);

	if(index > (base_value + num_cells - 1))
	{
		return nullptr;
	}
	else if(index == (base_value + num_cells - 1) && !node->has_children())
	{
		return node;
	}

	for(size_t child_index = 0; child_index < 8; ++child_index)
	{
		const grid_octree_node* child_node = node->get_child_at_index_const(child_index);
		size_t num_cells_child = cell_count_recursive(child_node);

		if(index >= base_value && index < base_value + num_cells_child)
		{
			return find_cell_by_index_recursive_const(child_node, index, base_value);
		}
		
		base_value += num_cells_child;
	}

	// Should never happen, but required by compiler.
	return nullptr;
}

const view_cell* grid_octree::
get_cell_at_position(const scm::math::vec3d& position, size_t* cell_index) const
{
	std::lock_guard<std::mutex> lock(mutex_);

	vec3r min_vertex(root_node_->get_position_center() - root_node_->get_size() * 0.5);
    vec3r max_vertex(root_node_->get_position_center() + root_node_->get_size() * 0.5);
    bounding_box root_node_bb(min_vertex, max_vertex);

    const view_cell* result_view_cell = nullptr;

    if(root_node_bb.contains(position))
    {
    	result_view_cell = this->find_cell_by_position_recursive_const(root_node_, position);

    	// Look for the index of the view cell if required by caller.
    	if(cell_index != nullptr)
	    {
	    	for(size_t tracked_cell_index = 0; tracked_cell_index < cells_by_indices_.size(); ++tracked_cell_index)
	    	{
	    		if(result_view_cell == cells_by_indices_[tracked_cell_index])
	    		{
	    			(*cell_index) = tracked_cell_index;
	    			break;
	    		}
	    	}
	    }
    }

	return result_view_cell;
}

grid_octree_node* grid_octree::
find_cell_by_position_recursive(grid_octree_node* node, const scm::math::vec3d& position) const
{
	//  Ugly again, for notes see above in find_cell_by_index_recursive.
	return const_cast<grid_octree_node*>(static_cast<const grid_octree&>(*this).find_cell_by_position_recursive_const(node, position));
}

const grid_octree_node* grid_octree::
find_cell_by_position_recursive_const(const grid_octree_node* node, const scm::math::vec3d& position) const
{
	vec3r min_vertex(node->get_position_center() - node->get_size() * 0.5);
    vec3r max_vertex(node->get_position_center() + node->get_size() * 0.5);
    bounding_box node_bb(min_vertex, max_vertex);

    if(node->has_children())
    {
    	for(size_t child_index = 0; child_index < 8; ++child_index)
    	{
    		const grid_octree_node* child_node = node->get_child_at_index_const(child_index);

    		vec3r min_vertex(child_node->get_position_center() - child_node->get_size() * 0.5);
    		vec3r max_vertex(child_node->get_position_center() + child_node->get_size() * 0.5);
    		bounding_box child_node_bb(min_vertex, max_vertex);

    		if(child_node_bb.contains(position))
		    {
		    	return find_cell_by_position_recursive_const(child_node, position);
		    }
    	}
    }

   	return node;
}

void grid_octree::
set_cell_visibility(const size_t& cell_index, const model_t& model_id, const node_t& node_id, const bool& visibility)
{
	// If this function is locked, high performance loss in the preprocessing will occur.
	//std::lock_guard<std::mutex> lock(mutex_);

	view_cell* view_node = cells_by_indices_[cell_index];
	view_node->set_visibility(model_id, node_id, visibility);
}

void grid_octree::
set_cell_visibility(const scm::math::vec3d& position, const model_t& model_id, const node_t& node_id, const bool& visibility)
{
	std::lock_guard<std::mutex> lock(mutex_);

	vec3r min_vertex(root_node_->get_position_center() - root_node_->get_size() * 0.5);
    vec3r max_vertex(root_node_->get_position_center() + root_node_->get_size() * 0.5);
    bounding_box root_node_bb(min_vertex, max_vertex);

    view_cell* result_view_cell = nullptr;

    if(root_node_bb.contains(position))
    {
    	result_view_cell = find_cell_by_position_recursive(root_node_, position);
    }

    if(result_view_cell != nullptr)
    {
    	result_view_cell->set_visibility(model_id, node_id, visibility);
    }
}


void grid_octree::
save_grid_to_file(const std::string& file_path) const
{
	save_octree_grid(file_path, get_grid_identifier());
}

void grid_octree::
save_octree_grid(const std::string& file_path, const std::string& grid_type) const
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

	// Root size and position
	file_out << root_node_->get_size().x << std::endl;
	file_out << root_node_->get_position_center().x << " " << root_node_->get_position_center().y << " " << root_node_->get_position_center().z << std::endl;

	// Write octree structure by writing split commands (0 = don't split, 1 = split).
	std::deque<const grid_octree_node*> unwritten_nodes;
	unwritten_nodes.push_back(root_node_);

	while(unwritten_nodes.size() != 0)
	{
		const grid_octree_node* top_node = unwritten_nodes[0];
		unwritten_nodes.pop_front();
		
		if(top_node->has_children())
		{
			file_out << '1';
			
			for(size_t child_index = 0; child_index < 8; ++child_index)
			{
				unwritten_nodes.push_back(top_node->get_child_at_index_const(child_index));
			}
		}
		else
		{
			file_out << '0';
		}
	}
	file_out << std::endl;

	// Save number of models, so we can later simply read the node numbers.
	file_out << ids_.size() << std::endl;

	// Save the number of node ids of each model.
	for(size_t model_index = 0; model_index < ids_.size(); ++model_index)
	{
		file_out << ids_[model_index] << " ";
	}

	file_out.close();
}

void grid_octree::
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
	size_t num_cells = cell_count_recursive(root_node_);
	for(size_t cell_index = 0; cell_index < num_cells; ++cell_index)
	{
		std::string current_cell_data = "";
		const view_cell* current_cell = cells_by_indices_[cell_index];

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
				if(current_cell->get_visibility(model_id, node_id))
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

bool grid_octree::
load_grid_from_file(const std::string& file_path)
{
	return load_octree_grid(file_path, get_grid_identifier());
}

bool grid_octree::
load_octree_grid(const std::string& file_path, const std::string& grid_type)
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
	std::string file_grid_type;
	file_in >> file_grid_type;
	if(file_grid_type != grid_type)
	{
		std::cout << "Wrong grid type: " << file_grid_type <<  " / " << grid_type << std::endl;
		return false;
	}

	// Read data to create root node.
	double root_size;
	file_in >> root_size;

	double pos_x, pos_y, pos_z;
	file_in >> pos_x >> pos_y >> pos_z;

	if(root_node_ != nullptr)
	{
		delete root_node_;
	}

	root_node_ = new grid_octree_node(root_size, scm::math::vec3d(pos_x, pos_y, pos_z));

	// Read and create octree structure.
	std::deque<grid_octree_node*> nodes_to_check;
	nodes_to_check.push_back(root_node_);

	while(nodes_to_check.size() != 0)
	{
		grid_octree_node* top_node = nodes_to_check[0];
		nodes_to_check.pop_front();
		
		char split_command;
		file_in >> split_command;

		if(split_command == '1')
		{
			top_node->split();
			
			for(size_t child_index = 0; child_index < 8; ++child_index)
			{
				nodes_to_check.push_back(top_node->get_child_at_index(child_index));
			}
		}
	}

	// Make sure view cells are accessible.
	compute_index_access();

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

bool grid_octree::
load_visibility_from_file(const std::string& file_path)
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_in;
	file_in.open(file_path, std::ios::in | std::ios::binary);

	if(!file_in.is_open())
	{
		return false;
	}

	size_t num_cells = cell_count_recursive(root_node_);
	for(size_t cell_index = 0; cell_index < num_cells; ++cell_index)
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

bool grid_octree::
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

void grid_octree::
clear_cell_visibility(const size_t& cell_index)
{
	if(this->get_num_models() <= cell_index)
	{
		return;
	}

	std::lock_guard<std::mutex> lock(mutex_);

	cells_by_indices_[cell_index]->clear_visibility_data();
}

model_t grid_octree::
get_num_models() const
{
	return ids_.size();
}

node_t grid_octree::
get_num_nodes(const model_t& model_id) const
{
	return ids_[model_id];
}

grid_octree_node* grid_octree::
get_root_node()
{
	return root_node_;
}

}
}