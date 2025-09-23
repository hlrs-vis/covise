// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_GRID_OCTREE_NODE_H
#define LAMURE_PVS_GRID_OCTREE_NODE_H

#include <vector>

#include <lamure/pvs/pvs.h>
#include <lamure/types.h>
#include "lamure/pvs/view_cell_regular.h"

namespace lamure
{
namespace pvs
{

class grid_octree_node : public view_cell_regular
{
public:
	grid_octree_node();
	grid_octree_node(const double& cell_size, const scm::math::vec3d& position_center);
	~grid_octree_node();

	virtual std::string get_cell_type() const;
	static std::string get_cell_identifier();

	virtual void split();
	virtual void collapse();

	virtual bool has_children() const;
	virtual grid_octree_node* get_child_at_index(const size_t& index);
	virtual const grid_octree_node* get_child_at_index_const(const size_t& index) const;

protected:
	grid_octree_node** child_nodes_;
};

}
}

#endif