// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/grid_optimizer_octree.h"

namespace lamure
{
namespace pvs
{

void grid_optimizer_octree::
optimize_grid(grid* input_grid, const float& equality_threshold)
{
	grid_octree* oct_grid = (grid_octree*)input_grid;

	check_and_optimize_node(oct_grid->get_root_node(), input_grid, equality_threshold);

	// Grid was most likely changed, so update the indices for fast access.
	oct_grid->compute_index_access();
}

bool grid_optimizer_octree::
check_and_optimize_node(grid_octree_node* node, grid* input_grid, const float& equality_threshold)
{
	bool change_detected = false;

	if(node->has_children())
	{
		unsigned int node_children_have_children = 0;
		for(int child_index = 0; child_index < 8; ++child_index)
		{
			grid_octree_node* child_node = node->get_child_at_index(child_index);
			if(child_node->has_children())
			{
				node_children_have_children++;
			}
		}

		if(node_children_have_children > 0)
		{
			// If one of the nodes has children, go one level deeper and try to find optimization entry point there.
			for(int child_index = 0; child_index < 8; ++child_index)
			{
				grid_octree_node* child_node = node->get_child_at_index(child_index);
				change_detected |= check_and_optimize_node(child_node, input_grid, equality_threshold);
			}

			if(change_detected)
			{
				// Re-count children having children.
				node_children_have_children = 0;
				for(int child_index = 0; child_index < 8; ++child_index)
				{
					grid_octree_node* child_node = node->get_child_at_index(child_index);
					if(child_node->has_children())
					{
						node_children_have_children++;
					}
				}

				// If the node doesn't have any children containing children, try collapsing it as well.
				if(node_children_have_children == 0)
				{
					try_collapse_node(node, input_grid, equality_threshold);
				}
			}
		}
		else
		{
			// Nodes do not go any deeper, so an optimization check is possible.
			change_detected = try_collapse_node(node, input_grid, equality_threshold);
		}
	}

	return change_detected;
}

bool grid_optimizer_octree::
try_collapse_node(grid_octree_node* node, grid* input_grid, const float& equality_threshold)
{
	if(node->has_children())
	{
		grid_octree_node tmp_node;

		// Collect visibility data of all child nodes.
		for(int child_index = 0; child_index < 8; ++child_index)
		{
			grid_octree_node* child_node = node->get_child_at_index(child_index);

			for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
			{
				for(node_t node_index = 0; node_index < input_grid->get_num_nodes(model_index); ++node_index)
				{
					if(child_node->get_visibility(model_index, node_index))
					{
						tmp_node.set_visibility(model_index, node_index, true);
					}
				}
			}
		}

		// Count entries of collected visibility.
		std::map<model_t, std::vector<node_t>> tmp_visibility = tmp_node.get_visible_indices();
		size_t num_visible_nodes = 0;

		for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
		{
			num_visible_nodes += tmp_visibility[model_index].size();
		}

		bool collapse = true;

		// Compare to each of the children.
		for(int child_index = 0; child_index < 8; ++child_index)
		{
			grid_octree_node* child_node = node->get_child_at_index(child_index);
			std::map<model_t, std::vector<node_t>> child_visibility = child_node->get_visible_indices();

			size_t num_visible_nodes_child = 0;
			for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
			{
				num_visible_nodes_child += child_visibility[model_index].size();
			}

			// Check if difference of visible nodes is within threshold.
			if((float)num_visible_nodes_child / (float)num_visible_nodes < equality_threshold)
			{
				collapse = false;
				break;
			}
		}

		if(collapse)
		{
			node->collapse();

			// Propagate visibility of child nodes to parent.
			for(model_t model_index = 0; model_index < input_grid->get_num_models(); ++model_index)
			{
				// Performance improving hack. Instantly allocates memory.
				node->set_visibility(model_index, input_grid->get_num_nodes(model_index)-1, false);

				for(node_t node_index = 0; node_index < input_grid->get_num_nodes(model_index); ++node_index)
				{
					node->set_visibility(model_index, node_index, tmp_node.get_visibility(model_index, node_index));
				}
			}

			return true;
		}
	}

	return false;
}

}
}