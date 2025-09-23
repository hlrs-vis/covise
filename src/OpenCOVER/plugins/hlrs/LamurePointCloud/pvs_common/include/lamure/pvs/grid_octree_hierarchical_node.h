// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_GRID_OCTREE_HIERARCHICAL_NODE_H
#define LAMURE_PVS_GRID_OCTREE_HIERARCHICAL_NODE_H

#include <vector>
#include <set>
#include <map>

#include <lamure/pvs/pvs.h>
#include <scm/core/math.h>
#include <lamure/types.h>
#include "lamure/pvs/grid_octree_node.h"

namespace lamure
{
namespace pvs
{

class grid_octree_hierarchical_node : public grid_octree_node
{
public:
	grid_octree_hierarchical_node();
	grid_octree_hierarchical_node(const double& cell_size, const scm::math::vec3d& position_center, grid_octree_hierarchical_node* parent);
	~grid_octree_hierarchical_node();

	virtual std::string get_cell_type() const;
	static std::string get_cell_identifier();

	virtual bool get_visibility(const model_t& object_id, const node_t& node_id) const;

	virtual std::map<model_t, std::vector<node_t>> get_visible_indices() const;

	virtual void split();

	void combine_visibility(const std::vector<node_t>& ids, const unsigned short& num_allowed_unequal_elements);
	void activate_hierarchical_mode(const bool& activate, const bool& propagate);

protected:
	const grid_octree_hierarchical_node* get_parent_node();

private:
	grid_octree_hierarchical_node* parent_;

	// If a grid is optimized, the information will be stored in a hierarchy. Access to visibility is different in that case.
	bool hierarchical_storage_;

};

}
}

#endif