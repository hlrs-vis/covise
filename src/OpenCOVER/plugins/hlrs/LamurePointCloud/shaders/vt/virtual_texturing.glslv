// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 440 core

uniform mat4 projection_matrix;
uniform mat4 model_view_matrix;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_texture_coord;
layout(location = 2) in vec3 in_normal;

out vec2 texture_coord;

//out vertex_data {
//    vec3 position;
//    vec3 normal;
//    vec2 coord;
//} vertex_out;


void main()
{
	//vertex_out.position = in_position;
    //vertex_out.normal = in_normal;
    //vertex_out.coord = in_texture_coord;
    texture_coord = in_texture_coord;

    gl_Position = projection_matrix * model_view_matrix * (vec4(in_position, 1.0));
}
