// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

//TODO: 3) write a very simple shader for rendering triangles (input: pos and uv), load the shader

#version 420 core

layout(location = 0) out vec4 out_color;
in vec2 texture_coord;


void main() {

    if (texture_coord.x <= 0.047619048 && texture_coord.y <= 0.047619048) {
        out_color = vec4(1,0,0,1);
    }
    else {
        out_color = vec4(0.f,texture_coord.x, texture_coord.y, 1.0f);
    }
}