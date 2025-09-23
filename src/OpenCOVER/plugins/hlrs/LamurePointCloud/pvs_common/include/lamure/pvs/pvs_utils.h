// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_PVSUTILS_H_
#define LAMURE_PVS_PVSUTILS_H_

#include <string>

#include <lamure/pvs/pvs.h>
#include "lamure/pvs/grid.h"

namespace lamure
{
namespace pvs
{

// Output data on the current visibility of a grid into a given file.
PVS_COMMON_DLL void analyze_grid_visibility(const grid* input_grid, const unsigned int& num_steps, const std::string& output_file_name);

// Calculate the current occlusion percentage within a given grid.
PVS_COMMON_DLL double calculate_grid_occlusion(const grid* input_grid);

}
}

#endif