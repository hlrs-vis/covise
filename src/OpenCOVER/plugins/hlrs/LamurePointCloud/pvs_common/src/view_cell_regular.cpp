// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/view_cell_regular.h"

namespace lamure
{
namespace pvs
{

view_cell_regular::
view_cell_regular()
{
	cell_size_ = 1.0;
	position_center_ = scm::math::vec3d(0.0, 0.0, 0.0);
}

view_cell_regular::
view_cell_regular(const double& cell_size, const scm::math::vec3d& position_center)
{
	cell_size_ = cell_size;
	position_center_ = position_center;
}

view_cell_regular::
~view_cell_regular()
{
}

std::string view_cell_regular::
get_cell_type() const
{
	return get_cell_identifier();
}

std::string view_cell_regular::
get_cell_identifier()
{
	return "view_cell_regular";
}

scm::math::vec3d view_cell_regular::
get_size() const
{
	return scm::math::vec3d(cell_size_, cell_size_, cell_size_);
}

scm::math::vec3d view_cell_regular::
get_position_center() const
{
	return position_center_;
}

void view_cell_regular::
set_visibility(const model_t& object_id, const node_t& node_id, const bool& visible)
{
	if(visibility_.size() <= object_id)
	{
		visibility_.resize(object_id + 1);
	}

	boost::dynamic_bitset<>& node_visibility = visibility_[object_id];

	if(node_visibility.size() <= node_id)
	{
		node_visibility.resize(node_id + 1);
	}

	node_visibility[node_id] = visible;
}

bool view_cell_regular::
get_visibility(const model_t& object_id, const node_t& node_id) const
{
	if(visibility_.size() <= object_id)
	{
		return false;
	}

	const boost::dynamic_bitset<>& node_visibility = visibility_[object_id];

	if(node_visibility.size() <= node_id)
	{
		return false;
	}

	return node_visibility[node_id];
}

bool view_cell_regular::
contains_visibility_data() const
{
	return visibility_.size() > 0;
}

std::map<model_t, std::vector<node_t>> view_cell_regular::
get_visible_indices() const
{
	std::map<model_t, std::vector<node_t>> indices;

	for(model_t model_index = 0; model_index < visibility_.size(); ++model_index)
	{
		const boost::dynamic_bitset<>& node_visibility = visibility_[model_index];

		for(node_t node_index = 0; node_index < node_visibility.size(); ++node_index)
		{
			if(node_visibility[node_index])
			{
				indices[model_index].push_back(node_index);
			}
		}
	}

	return indices;
}

void view_cell_regular::
clear_visibility_data()
{
	visibility_.clear();
}

boost::dynamic_bitset<> view_cell_regular::
get_bitset(const model_t& object_id) const
{
	if(visibility_.size() <= object_id)
	{
		return boost::dynamic_bitset<>();
	}

	return visibility_[object_id];
}

void view_cell_regular::
set_bitset(const model_t& object_id, const boost::dynamic_bitset<>& bitset)
{
	if(visibility_.size() <= object_id)
	{
		visibility_.resize(object_id + 1);
	}

	visibility_[object_id] = boost::dynamic_bitset<>(bitset);
}

}
}
