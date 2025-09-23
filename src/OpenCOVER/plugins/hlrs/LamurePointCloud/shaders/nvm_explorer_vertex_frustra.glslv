// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core
  
layout (location = 0) in vec3 position;

uniform mat4 matrix_model;
uniform mat4 matrix_view;
uniform mat4 matrix_perspective;

void main()
{
  // if(is_highlighted > 0.5)
  // {
    gl_PointSize = 5.0;
  // } else {
  // }
    //gl_Position = vec4(position.x, position.y, position.z, 1.0);
  // passed_color = color;
  // passed_is_highlighted = is_highlighted;
  // passed_position_view_space = matrix_view * vec4(position, 1.0);
  // gl_Position = vec4(position.xy, 0.1, 1.0);
  // gl_Position = matrix_perspective * matrix_view * vec4(position, 1.0);
  gl_Position = matrix_perspective * matrix_view * matrix_model * vec4(position, 1.0);
}