// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

uniform bool has_pixels;

out vec4 color;

void main()
{
    if(has_pixels)
    {
        color = vec4(0.0, 1.0, 1.0, 1.0f);
    }
    else
    {
        color = vec4(0.0, 1.0, 0.0, 1.0f);
    }
}