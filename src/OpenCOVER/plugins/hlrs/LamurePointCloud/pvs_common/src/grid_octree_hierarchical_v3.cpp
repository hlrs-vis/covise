// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <fstream>
#include <deque>

#include "lamure/pvs/grid_octree_hierarchical_v3.h"

namespace lamure
{
namespace pvs
{

grid_octree_hierarchical_v3::
grid_octree_hierarchical_v3() : grid_octree_hierarchical_v3(1, 1.0, scm::math::vec3d(0.0, 0.0, 0.0), std::vector<node_t>())
{
}

grid_octree_hierarchical_v3::
grid_octree_hierarchical_v3(const size_t& octree_depth, const double& size, const scm::math::vec3d& position_center, const std::vector<node_t>& ids) : grid_octree_hierarchical(octree_depth, size, position_center, ids)
{
}

grid_octree_hierarchical_v3::
~grid_octree_hierarchical_v3()
{
}

std::string grid_octree_hierarchical_v3::
get_grid_type() const
{
	return get_grid_identifier();
}

std::string grid_octree_hierarchical_v3::
get_grid_identifier()
{
	return "octree_hierarchical_v3";
}

void grid_octree_hierarchical_v3::
save_grid_to_file(const std::string& file_path) const
{
	save_octree_grid(file_path, get_grid_identifier());
}

void grid_octree_hierarchical_v3::
save_visibility_to_file(const std::string& file_path) const
{
	// When saving, the data should only be read from the single nodes, not collected from their parents.
	((grid_octree_hierarchical_node*)root_node_)->activate_hierarchical_mode(false, true);

	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_out;
	file_out.open(file_path, std::ios::out | std::ios::binary);

	if(!file_out.is_open())
	{
		throw std::invalid_argument("invalid file path: " + file_path);
	}

	// Iterate over view cells.
	std::deque<const grid_octree_node*> unwritten_nodes;
	unwritten_nodes.push_back(root_node_);

	while(unwritten_nodes.size() != 0)
	{
		const grid_octree_hierarchical_node* current_node = (const grid_octree_hierarchical_node*)unwritten_nodes[0];
		unwritten_nodes.pop_front();

		for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
		{
			std::vector<node_t> model_visibility;

			// Collect the visibility.
			for(node_t node_index = 0; node_index < ids_[model_index]; ++node_index)
			{
				if(current_node->get_visibility(model_index, node_index))
				{
					model_visibility.push_back(node_index);
				}
			}

			// Write number of values which will be written so it can be read later.
			node_t number_visibility_elements = model_visibility.size();
			file_out.write(reinterpret_cast<char*>(&number_visibility_elements), sizeof(number_visibility_elements));

			// Write visibility data.
			for(node_t node_index = 0; node_index < number_visibility_elements; ++node_index)
			{
				node_t visibility_node = model_visibility[node_index];
				file_out.write(reinterpret_cast<char*>(&visibility_node), sizeof(visibility_node));
			}
		}

		// Add child nodes of current nodes to queue.
		if(current_node->has_children())
		{
			for(size_t child_index = 0; child_index < 8; ++child_index)
			{
				unwritten_nodes.push_back(current_node->get_child_at_index_const(child_index));
			}
		}
	}

	file_out.close();
}

bool grid_octree_hierarchical_v3::
load_grid_from_file(const std::string& file_path)
{
	return load_octree_grid(file_path, get_grid_identifier());
}


bool grid_octree_hierarchical_v3::
load_visibility_from_file(const std::string& file_path)
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_in;
	file_in.open(file_path, std::ios::in | std::ios::binary);

	if(!file_in.is_open())
	{
		return false;
	}

	// Iterate over view cells.
	std::deque<grid_octree_node*> unread_nodes;
	unread_nodes.push_back(root_node_);

	while(unread_nodes.size() != 0)
	{
		grid_octree_hierarchical_node* current_node = (grid_octree_hierarchical_node*)unread_nodes[0];
		unread_nodes.pop_front();

		for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
		{
			// Set all nodes to proper default value depending on which kind of visibility index is stored.
			for(node_t node_index = 0; node_index < ids_[model_index]; ++node_index)
			{
				current_node->set_visibility(model_index, node_index, false);
			}

			// Read number of values to be read afterwards.
			node_t number_visibility_elements;
			file_in.read(reinterpret_cast<char*>(&number_visibility_elements), sizeof(number_visibility_elements));

			for(node_t node_index = 0; node_index < number_visibility_elements; ++node_index)
			{
				node_t visibility_index;
				file_in.read(reinterpret_cast<char*>(&visibility_index), sizeof(visibility_index));
				current_node->set_visibility(model_index, visibility_index, true);
			}
		}

		// Add child nodes of current nodes to queue.
		if(current_node->has_children())
		{
			for(size_t child_index = 0; child_index < 8; ++child_index)
			{
				unread_nodes.push_back(current_node->get_child_at_index(child_index));
			}
		}

		propagate_node_visibility(current_node);
	}

	file_in.close();
	return true;
}

bool grid_octree_hierarchical_v3::
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

	// Must iterate over whole parent nodes from root to leaf level first.
	size_t num_parents = 0;
	size_t current_parent_level = cells_by_indices_.size();

	// Sum up parent levels.
	while(current_parent_level / 8 > 0)
	{
		current_parent_level = current_parent_level / 8;
		num_parents += current_parent_level;
	}

	// Move to proper start point within file.
	for(size_t jump_over_index = 0; jump_over_index < (cell_index + num_parents); ++jump_over_index)
	{	
		for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
		{
			// Read number of values to pass.
			node_t number_visibility_elements;
			file_in.read(reinterpret_cast<char*>(&number_visibility_elements), sizeof(number_visibility_elements));

			file_in.seekg(file_in.tellg() + (std::streampos)(number_visibility_elements * sizeof(number_visibility_elements)));
		}
	}

	// Read the visibility data.
	for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
	{
		// Set all nodes to proper default value depending on which kind of visibility index is stored.
		for(node_t node_index = 0; node_index < ids_[model_index]; ++node_index)
		{
			current_cell->set_visibility(model_index, node_index, false);
		}

		// Read number of values to be read afterwards.
		node_t number_visibility_elements;
		file_in.read(reinterpret_cast<char*>(&number_visibility_elements), sizeof(number_visibility_elements));

		// Read visible IDs.
		for(node_t node_index = 0; node_index < number_visibility_elements; ++node_index)
		{
			node_t visibility_index;
			file_in.read(reinterpret_cast<char*>(&visibility_index), sizeof(visibility_index));
			current_cell->set_visibility(model_index, visibility_index, true);

		}
		
		propagate_node_visibility(current_cell);
	}

	file_in.close();
	return true;
}

void grid_octree_hierarchical_v3::
propagate_node_visibility(view_cell* cell)
{
	std::map<model_t, std::vector<node_t>> visible_indices = cell->get_visible_indices();

	for(std::map<model_t, std::vector<node_t>>::const_iterator map_iter = visible_indices.begin(); map_iter != visible_indices.end(); ++map_iter)
    {
        for(node_t node_index = 0; node_index < map_iter->second.size(); ++node_index)
        {
            node_t visible_node_id = map_iter->second.at(node_index);

            // Communicate visibility to children and parents nodes of visible nodes.
            set_node_children_visible(cell, map_iter->first, visible_node_id);
            set_node_parents_visible(cell, map_iter->first, visible_node_id);
        }       
    }
}

void grid_octree_hierarchical_v3::
set_node_parents_visible(view_cell* cell, const model_t& model_id, const node_t& node_id)
{
	if(node_id == 0)
	{
		return;
	}

	// Note how the fan factor 2 is an assumption.
	node_t fan_factor = 2;

	// No access no actual model data, so ID has to be calculated locally.
	node_t parent_id = 0;
    if (node_id % fan_factor == 0)
    {
        parent_id = node_id/fan_factor - 1;
    }
    else
    {
        parent_id = (node_id + fan_factor - (node_id % fan_factor)) / fan_factor - 1;
    }
    
    // Apply visibility propagation and continue upwards.
    if(parent_id != lamure::invalid_node_t && !cell->get_visibility(model_id, parent_id))
    {
        cell->set_visibility(model_id, parent_id, true);
        set_node_parents_visible(cell, model_id, parent_id);
    }
}

void grid_octree_hierarchical_v3::
set_node_children_visible(view_cell* cell, const model_t& model_id, const node_t& node_id)
{
	// Note how the fan factor 2 is an assumption.
	node_t fan_factor = 2;

    for(uint32_t child_index = 0; child_index < fan_factor; ++child_index)
    {
    	// No access no actual model data, so ID has to be calculated locally.
		node_t child_id =  node_id * fan_factor + 1 + child_index;

        if(child_id < ids_[model_id] && !cell->get_visibility(model_id, child_id))
        {
            cell->set_visibility(model_id, child_id, true);
            set_node_children_visible(cell, model_id, child_id);
        }
    }
}

}
}