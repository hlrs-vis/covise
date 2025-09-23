// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

// in vec4 passed_position_view_space;
// in vec3 passed_color;
// in float passed_is_highlighted;
uniform bool seen;

out vec4 color;

void main()
{
    if(seen)
    {
        color = vec4(1.0, 0.0, 0.0, 1.0f);
    }
    else
    {
        color = vec4(1.0, 0.8, 0.0, 1.0f);
    }
}