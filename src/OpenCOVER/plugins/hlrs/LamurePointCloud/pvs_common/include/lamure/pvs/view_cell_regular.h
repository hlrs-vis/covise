// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_VIEW_CELL_REGULAR_H
#define LAMURE_PVS_VIEW_CELL_REGULAR_H

#include <vector>
#include <map>

#include <scm/core/math.h>
#include <lamure/types.h>

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/view_cell.h"

#include <boost/dynamic_bitset.hpp>

namespace lamure
{
namespace pvs
{

class view_cell_regular : public view_cell
{
public:
	view_cell_regular();
	view_cell_regular(const double& cell_size, const scm::math::vec3d& position_center);
	~view_cell_regular();

	virtual std::string get_cell_type() const;
	static std::string get_cell_identifier();

	virtual scm::math::vec3d get_size() const;
	virtual scm::math::vec3d get_position_center() const;

	virtual void set_visibility(const model_t& object_id, const node_t& node_id, const bool& visible);
	virtual bool get_visibility(const model_t& object_id, const node_t& node_id) const;

	virtual bool contains_visibility_data() const;
	virtual std::map<model_t, std::vector<node_t>> get_visible_indices() const;
	virtual void clear_visibility_data();

	virtual boost::dynamic_bitset<> get_bitset(const model_t& object_id) const;
	virtual void set_bitset(const model_t& object_id, const boost::dynamic_bitset<>& bitset);

private:
	double cell_size_;
	scm::math::vec3d position_center_;

	std::vector<boost::dynamic_bitset<>> visibility_;
};

}
}

#endif
