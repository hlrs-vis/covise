// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

layout (location = 0) in vec4 vertex_position;
layout (location = 1) in vec2 vertex_coord;
layout (location = 2) in vec4 vertex_normal;

uniform mat4 mvp_matrix;

out vertex_data {
    vec4 position;
    vec4 normal;
    vec2 coord;
} vertex_out;


void main() {

  vertex_out.position = vertex_position;
  vertex_out.normal = vertex_normal;
  vertex_out.coord = vertex_coord;
  gl_Position = mvp_matrix * vertex_position;
}
