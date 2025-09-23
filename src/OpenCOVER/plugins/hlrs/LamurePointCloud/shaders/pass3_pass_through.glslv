// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

layout(location = 0) in vec3 in_position;

out vec2 pos;

void main()
{
     gl_Position = vec4(in_position, 1.0);

     pos = in_position.xy;
}
