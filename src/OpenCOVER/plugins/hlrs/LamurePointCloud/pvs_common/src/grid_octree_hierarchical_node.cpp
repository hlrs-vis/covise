// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/grid_octree_hierarchical_node.h"

namespace lamure
{
namespace pvs
{

grid_octree_hierarchical_node::
grid_octree_hierarchical_node() : grid_octree_node()
{
	parent_ = nullptr;
	hierarchical_storage_ = false;
}

grid_octree_hierarchical_node::
grid_octree_hierarchical_node(const double& cell_size, const scm::math::vec3d& position_center, grid_octree_hierarchical_node* parent) : grid_octree_node(cell_size, position_center)
{
	parent_ = parent;
	hierarchical_storage_ = false;
}

grid_octree_hierarchical_node::
~grid_octree_hierarchical_node()
{
}

std::string grid_octree_hierarchical_node::
get_cell_type() const
{
	return get_cell_identifier();
}

std::string grid_octree_hierarchical_node::
get_cell_identifier()
{
	return "grid_octree_hierarchical_node";
}

bool grid_octree_hierarchical_node::
get_visibility(const model_t& object_id, const node_t& node_id) const
{
	bool visible = grid_octree_node::get_visibility(object_id, node_id);
	
	// If node is not visible in current node, it might still be in a parent node.
	if(hierarchical_storage_ && !visible && parent_ != nullptr)
	{
		visible = parent_->get_visibility(object_id, node_id);
	}

	return visible;
}

std::map<model_t, std::vector<node_t>> grid_octree_hierarchical_node::
get_visible_indices() const
{
	std::map<model_t, std::vector<node_t>> indices = grid_octree_node::get_visible_indices();

	// If visibility is stored hierarchically, parent visibility must be included as well.
	if(hierarchical_storage_ && parent_ != nullptr)
	{
		std::map<model_t, std::vector<node_t>> parent_indices = parent_->get_visible_indices();

		for(std::map<model_t, std::vector<node_t>>::iterator iter = parent_indices.begin(); iter != parent_indices.end(); ++iter)
		{
			const model_t& model_key = iter->first;
			std::vector<node_t>& old_nodes = indices[model_key];
			std::vector<node_t>& new_nodes = iter->second;

			for(node_t node_index = 0; node_index < new_nodes.size(); ++node_index)
			{
				// For this to work, we assume the code works and the parents do not include any redundant values.
				old_nodes.push_back(new_nodes[node_index]);
			}
		}
	}

	return indices;
}

void grid_octree_hierarchical_node::
split()
{
	if(child_nodes_ == nullptr)
	{
		child_nodes_ = new grid_octree_node*[8];
		
		for(size_t child_index = 0; child_index < 8; ++child_index)
		{
			scm::math::vec3d new_pos = get_position_center();
			
			scm::math::vec3d multiplier(1.0, 1.0, 1.0);

			if(child_index % 2 == 0)
			{
				multiplier.x = -1.0;
			}

			if(child_index / 4 == 1)
			{
				multiplier.y = -1.0;
			}

			if(child_index % 4 >= 2)
			{
				multiplier.z = -1.0;
			}

			double new_size = get_size().x * 0.5;
			new_pos = new_pos + (multiplier * get_size().x * 0.25);

			child_nodes_[child_index] = new grid_octree_hierarchical_node(new_size, new_pos, this);
		}
	}
}

void grid_octree_hierarchical_node::
combine_visibility(const std::vector<node_t>& ids, const unsigned short& num_allowed_unequal_elements)
{
	hierarchical_storage_ = true;

	if(this->has_children())
	{
		// Make sure all children have been processed recursively.
		for(size_t child_index = 0; child_index < 8; ++child_index)
		{
			grid_octree_hierarchical_node* current_child_node = (grid_octree_hierarchical_node*)this->get_child_at_index(child_index);
			current_child_node->combine_visibility(ids, num_allowed_unequal_elements);
		}

		for(model_t model_index = 0; model_index < ids.size(); ++model_index)
		{
			for(node_t node_index = 0; node_index < ids[model_index]; ++node_index)
			{
				unsigned short non_appearance_counter = 0;

				// Count how often the index doesn't appear (allows faster skip of the current ID).
				for(size_t child_index = 0; child_index < 8; ++child_index)
				{
					grid_octree_node* current_child_node = this->get_child_at_index(child_index);

					if(!current_child_node->get_visibility(model_index, node_index))
					{
						non_appearance_counter++;

						if(non_appearance_counter > num_allowed_unequal_elements)
						{
							break;
						}
					}
				}

				// If an element is common among all children (or a given threshold of children) it is moved to the parent.
				if(non_appearance_counter <= num_allowed_unequal_elements)
				{
					this->set_visibility(model_index, node_index, true);

					for(size_t child_index = 0; child_index < 8; ++child_index)
					{
						this->get_child_at_index(child_index)->set_visibility(model_index, node_index, false);
					}
				}
			}
		}
	}
}

void grid_octree_hierarchical_node::
activate_hierarchical_mode(const bool& activate, const bool& propagate)
{
	hierarchical_storage_ = activate;

	if(propagate)
	{
		if(this->has_children())
		{
			for(size_t child_index = 0; child_index < 8; ++child_index)
			{
				grid_octree_hierarchical_node* child_node = (grid_octree_hierarchical_node*)this->get_child_at_index(child_index);
				child_node->activate_hierarchical_mode(activate, propagate);
			}
		}
	}
}

const grid_octree_hierarchical_node* grid_octree_hierarchical_node::
get_parent_node()
{
	return parent_;
}

}
}
