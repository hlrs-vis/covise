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

out vec4 color;

void main()
{
    // if(passed_is_highlighted > 0.5)
    // {

    color = vec4(0.0, 0.0, 1.0, 1.0f);
    // } else {
    // color = vec4(passed_color, 1.0f);
    // }
}