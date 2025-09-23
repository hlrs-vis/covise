// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_VIEW_CELL_IRREGULAR_MANAGING_H
#define LAMURE_PVS_VIEW_CELL_IRREGULAR_MANAGING_H

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/view_cell_regular.h"

namespace lamure
{
namespace pvs
{

class view_cell_regular_managing : public view_cell_regular
{
public:
	view_cell_regular_managing();
	view_cell_regular_managing(const double& cell_size, const scm::math::vec3d& position_center);
	~view_cell_regular_managing();

	virtual std::string get_cell_type() const;
	static std::string get_cell_identifier();

	virtual scm::math::vec3d get_size() const;
	virtual scm::math::vec3d get_position_center() const;

	void add_cell(const view_cell* cell);

	void set_error(const float& error);
	float get_error() const;

private:
	std::vector<const view_cell*> managed_view_cells_;
	float current_error_;
};

}
}

#endif
