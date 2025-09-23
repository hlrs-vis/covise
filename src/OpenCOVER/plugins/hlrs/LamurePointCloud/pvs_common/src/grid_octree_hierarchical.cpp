// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <fstream>
#include <deque>

#include "lamure/pvs/grid_octree_hierarchical.h"
#include "lamure/pvs/pvs_utils.h"

namespace lamure
{
namespace pvs
{

grid_octree_hierarchical::
grid_octree_hierarchical() : grid_octree_hierarchical(1, 1.0, scm::math::vec3d(0.0, 0.0, 0.0), std::vector<node_t>())
{
}

grid_octree_hierarchical::
grid_octree_hierarchical(const size_t& octree_depth, const double& size, const scm::math::vec3d& position_center, const std::vector<node_t>& ids)
{
	root_node_ = new grid_octree_hierarchical_node(size, position_center, nullptr);
	
	create_grid(root_node_, octree_depth);
	
	compute_index_access();

	ids_.resize(ids.size());
	for(size_t index = 0; index < ids_.size(); ++index)
	{
		ids_[index] = ids[index];
	}
}

grid_octree_hierarchical::
~grid_octree_hierarchical()
{
	if(root_node_ != nullptr)
	{
		delete root_node_;
	}
}

std::string grid_octree_hierarchical::
get_grid_type() const
{
	return get_grid_identifier();
}

std::string grid_octree_hierarchical::
get_grid_identifier()
{
	return "octree_hierarchical";
}

void grid_octree_hierarchical::
save_grid_to_file(const std::string& file_path) const
{
	save_octree_grid(file_path, get_grid_identifier());
}

void grid_octree_hierarchical::
save_visibility_to_file(const std::string& file_path) const
{
	// When saving, the data should only be read from the single nodes, nod collect from their parents.
	((grid_octree_hierarchical_node*)root_node_)->activate_hierarchical_mode(false, true);

	// If the occlusion is below 50%, save the indices of occluded nodes instead of visible nodes to save storage.
	bool save_occlusion = calculate_average_node_hierarchy_visibility() > 0.5;

	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_out;
	file_out.open(file_path, std::ios::out | std::ios::binary);

	if(!file_out.is_open())
	{
		throw std::invalid_argument("invalid file path: " + file_path);
	}

	// First byte is used to save whether visibility or occlusion data are written.
	file_out.write(reinterpret_cast<char*>(&save_occlusion), sizeof(save_occlusion));	

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

			// Collect the visibility nodes depending on the occlusion situation.
			for(node_t node_index = 0; node_index < ids_[model_index]; ++node_index)
			{
				if(current_node->get_visibility(model_index, node_index) != save_occlusion)
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

bool grid_octree_hierarchical::
load_grid_from_file(const std::string& file_path)
{
	return load_octree_grid(file_path, get_grid_identifier());
}

bool grid_octree_hierarchical::
load_visibility_from_file(const std::string& file_path)
{
	std::lock_guard<std::mutex> lock(mutex_);

	std::fstream file_in;
	file_in.open(file_path, std::ios::in | std::ios::binary);

	if(!file_in.is_open())
	{
		return false;
	}

	// First byte is used to save whether visibility or occlusion data are written.
	bool load_occlusion;
	file_in.read(reinterpret_cast<char*>(&load_occlusion), sizeof(load_occlusion));

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
				current_node->set_visibility(model_index, node_index, load_occlusion);
			}

			// Read number of values to be read afterwards.
			node_t number_visibility_elements;
			file_in.read(reinterpret_cast<char*>(&number_visibility_elements), sizeof(number_visibility_elements));

			for(node_t node_index = 0; node_index < number_visibility_elements; ++node_index)
			{
				node_t visibility_index;
				file_in.read(reinterpret_cast<char*>(&visibility_index), sizeof(visibility_index));
				current_node->set_visibility(model_index, visibility_index, !load_occlusion);
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
	}

	// Iterate over view cells.
	std::deque<grid_octree_node*> unpropagated_nodes;
	unpropagated_nodes.push_back(root_node_);

	while(unpropagated_nodes.size() != 0)
	{
		grid_octree_hierarchical_node* current_node = (grid_octree_hierarchical_node*)unpropagated_nodes[0];
		unpropagated_nodes.pop_front();

		if(current_node->has_children())
		{
			for(size_t child_index = 0; child_index < 8; ++child_index)
			{
				grid_octree_node* child_node = current_node->get_child_at_index(child_index);
				std::map<model_t, std::vector<node_t>> visible_indices = current_node->get_visible_indices();

				for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
				{
					const std::vector<node_t>& model_visibility = visible_indices[model_index];

					for(node_t node_index = 0; node_index < model_visibility.size(); ++node_index)
					{
						child_node->set_visibility(model_index, model_visibility[node_index], true);
					}
				}

				// Add child nodes of current nodes to queue.
				unpropagated_nodes.push_back(child_node);
			}

			// Clear node visibility since it has been propagated to children.
			current_node->clear_visibility_data();
		}
	}

	file_in.close();
	return true;
}

bool grid_octree_hierarchical::
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

	// First byte is used to save whether visibility or occlusion data are written.
	bool load_occlusion;
	file_in.read(reinterpret_cast<char*>(&load_occlusion), sizeof(load_occlusion));

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

	// WARNING: currently completely ignoring parent visibility! Therefore, runtime access is not correct!
	// TODO: implement proper access including parents.

	// Read the visibility data.
	for(model_t model_index = 0; model_index < ids_.size(); ++model_index)
	{
		// Set all nodes to proper default value depending on which kind of visibility index is stored.
		for(node_t node_index = 0; node_index < ids_[model_index]; ++node_index)
		{
			current_cell->set_visibility(model_index, node_index, load_occlusion);
		}

		// Read number of values to be read afterwards.
		node_t number_visibility_elements;
		file_in.read(reinterpret_cast<char*>(&number_visibility_elements), sizeof(number_visibility_elements));

		for(node_t node_index = 0; node_index < number_visibility_elements; ++node_index)
		{
			node_t visibility_index;
			file_in.read(reinterpret_cast<char*>(&visibility_index), sizeof(visibility_index));
			current_cell->set_visibility(model_index, visibility_index, !load_occlusion);
		}
	}

	file_in.close();
	return true;
}

void grid_octree_hierarchical::
combine_visibility(const unsigned short& num_allowed_unequal_elements)
{
	grid_octree_hierarchical_node* root_hierarchical_node = (grid_octree_hierarchical_node*)root_node_;
	root_hierarchical_node->combine_visibility(ids_, num_allowed_unequal_elements);
}

double grid_octree_hierarchical::
calculate_average_node_hierarchy_visibility() const
{
	// Iterate over nodes and collect visibility data.
    size_t total_num_nodes = 0;
    size_t total_visible_nodes = 0;

    // Iterate over view cells.
	std::deque<const grid_octree_node*> unchecked_nodes;
	unchecked_nodes.push_back(root_node_);

	while(unchecked_nodes.size() != 0)
	{
		grid_octree_hierarchical_node* current_node = (grid_octree_hierarchical_node*)unchecked_nodes[0];
		unchecked_nodes.pop_front();

		lamure::model_t num_models = this->get_num_models();
       	std::map<lamure::model_t, std::vector<lamure::node_t>> visibility = current_node->get_visible_indices();

		lamure::node_t model_num_nodes = 0;
        lamure::node_t model_visible_nodes = 0;

        for(lamure::model_t model_index = 0; model_index < num_models; ++model_index)
        {
            lamure::node_t num_nodes = this->get_num_nodes(model_index);

            model_num_nodes += num_nodes;
            model_visible_nodes += visibility[model_index].size();
        }

        total_num_nodes += model_num_nodes;
        total_visible_nodes += model_visible_nodes;

        // Add child nodes of current nodes to queue.
		if(current_node->has_children())
		{
			for(size_t child_index = 0; child_index < 8; ++child_index)
			{
				unchecked_nodes.push_back(current_node->get_child_at_index_const(child_index));
			}
		}
	}

    return (double)total_visible_nodes / (double)total_num_nodes;
}

}
}