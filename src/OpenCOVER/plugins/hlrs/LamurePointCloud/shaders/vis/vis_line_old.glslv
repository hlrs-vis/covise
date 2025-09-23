// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

uniform mat4 projection_matrix;
uniform mat4 view_matrix;

layout(location = 0) in vec3 in_position;

void main() {
  gl_Position = projection_matrix * view_matrix * vec4(in_position, 1.0);
}
