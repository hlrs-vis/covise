// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

layout(location = 0) in vec3 position;

uniform mat4 matrix_view;
uniform mat4 matrix_perspective;

void main() { gl_Position = matrix_perspective * matrix_view * vec4(position, 1.0); }