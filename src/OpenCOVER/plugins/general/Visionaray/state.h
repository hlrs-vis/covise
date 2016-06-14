/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifndef VSNRAY_COVER_STATE_H
#define VSNRAY_COVER_STATE_H 1

#include <visionaray/detail/call_kernel.h> // visionaray::detail::algorithm

namespace visionaray { namespace cover {


//-------------------------------------------------------------------------------------------------
// Control scene update behavior
//

enum device_type    { CPU, GPU };
enum data_variance  { Static, Dynamic };
enum color_space    { RGB, sRGB };


//-------------------------------------------------------------------------------------------------
// State that affects rendering of a frame
//

struct render_state
{
    detail::algorithm   algo            = detail::Simple;
    unsigned            min_bounces     = 1;
    unsigned            max_bounces     = 10;
    unsigned            num_bounces     = 4;
    device_type         device          = CPU;
    data_variance       data_var        = Static;
    color_space         clr_space       = sRGB;
    unsigned            num_threads     = 0;
};


//-------------------------------------------------------------------------------------------------
// State that controls debug features
//

struct debug_state
{
    bool                debug_mode      = true;
    bool                show_bvh        = false;
    bool                show_bvh_costs  = false;
    bool                show_normals    = false;
    bool                show_tex_coords = false;
};

}} // namespace visionaray::cover

#endif // VSNRAY_COVER_STATE_H
