// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_GRID_OCTREE_HIERARCHICAL_H
#define LAMURE_PVS_GRID_OCTREE_HIERARCHICAL_H

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/grid_octree.h"
#include "lamure/pvs/grid_octree_node.h"
#include "lamure/pvs/grid_octree_hierarchical_node.h"

#include <mutex>

namespace lamure
{
namespace pvs
{

// Grid that stores the visibility as uint32 IDs using the octree-hierarchy.
// Common visibility will be stored within parent nodes instead of the children to save storage.
// Decides to save either visible or occluded IDs for the whole file.
class grid_octree_hierarchical : public grid_octree
{
public:
	grid_octree_hierarchical();
	grid_octree_hierarchical(const size_t& octree_depth, const double& size, const scm::math::vec3d& position_center, const std::vector<node_t>& ids);
	virtual ~grid_octree_hierarchical();

	virtual std::string get_grid_type() const;
	static std::string get_grid_identifier();

	virtual void save_grid_to_file(const std::string& file_path) const;
	virtual void save_visibility_to_file(const std::string& file_path) const;

	virtual bool load_grid_from_file(const std::string& file_path);
	virtual bool load_visibility_from_file(const std::string& file_path);

	virtual bool load_cell_visibility_from_file(const std::string& file_path, const size_t& cell_index);

	void combine_visibility(const unsigned short& num_allowed_unequal_elements);

protected:
	double calculate_average_node_hierarchy_visibility() const;
};

}
}

#endif
