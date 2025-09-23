// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

layout(location = 0) in vec3 position;
// layout (location = 1) in vec3 color;
// layout (location = 2) in float is_highlighted;

uniform mat4 matrix_view;
uniform mat4 matrix_perspective;
uniform int active_camera;

// out vec4 passed_position_view_space;
// out vec3 passed_color;
// out float passed_is_highlighted;

void main()
{
    // if(is_highlighted > 0.5)
    // {
    //   gl_PointSize = 5.0;
    // } else {
    if(active_camera == gl_VertexID)
    {
        gl_PointSize = 15;
    }
    else
    {
        gl_PointSize = 5;
    }
    // }
    // gl_Position = vec4(position.x, position.y, position.z, 1.0);

    // passed_color = color;
    // passed_is_highlighted = is_highlighted;
    // passed_position_view_space = matrix_view * vec4(position, 1.0);
    gl_Position = matrix_perspective * matrix_view * vec4(position, 1.0);
}