// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "lamure/pvs/view_cell_irregular.h"

namespace lamure
{
namespace pvs
{

view_cell_irregular::
view_cell_irregular() : view_cell_irregular(scm::math::vec3d(1.0, 1.0, 1.0), scm::math::vec3d(0.0, 0.0, 0.0))
{
}

view_cell_irregular::
view_cell_irregular(const scm::math::vec3d& cell_size, const scm::math::vec3d& position_center) : view_cell_regular(0.0, position_center)
{
	cell_size_ = cell_size;
}

view_cell_irregular::
~view_cell_irregular()
{
}

std::string view_cell_irregular::
get_cell_type() const
{
	return get_cell_identifier();
}

std::string view_cell_irregular::
get_cell_identifier()
{
	return "view_cell_irregular";
}

scm::math::vec3d view_cell_irregular::
get_size() const
{
	return cell_size_;
}

}
}