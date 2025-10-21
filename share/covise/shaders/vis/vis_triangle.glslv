// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

//TODO: 3) write a very simple shader for rendering triangles (input: pos and uv), load the shader
#version 420 core

uniform mat4 projection_matrix;
uniform mat4 view_matrix;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_texture_coord;

out vec2 texture_coord;


void main() {
  gl_Position = projection_matrix * view_matrix * vec4(in_position, 1.0);
  texture_coord = in_texture_coord;
}