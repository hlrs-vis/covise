// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_BOUNDS_EXTRAPOLATOR_H
#define LAMURE_PVS_BOUNDS_EXTRAPOLATOR_H

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/grid.h"
#include "lamure/pvs/grid_bounding.h"

namespace lamure
{
namespace pvs
{

class pvs_bounds_extrapolator
{
public:
	virtual ~pvs_bounds_extrapolator() {}

	virtual grid_bounding* extrapolate_from_grid(const grid* input_grid) const = 0;
};

}
}

#endif
