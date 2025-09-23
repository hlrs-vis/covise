// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

layout (location = 0) out vec4 out_color;

in vertex_data {
    vec4 position;
    vec4 normal;
    vec2 coord;
} vertex_in;

void main()
{
  vec4 n = vertex_in.normal;
  out_color = vec4(n.r*0.5+0.5, n.g*0.5+0.5, n.b*0.5+0.5, 1.0f);
}

