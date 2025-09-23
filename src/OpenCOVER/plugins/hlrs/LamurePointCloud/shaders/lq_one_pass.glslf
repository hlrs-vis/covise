// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

uniform bool state_lense;
uniform int width_window;
uniform int height_window;
uniform float radius_sphere_screen;
uniform vec2 position_sphere_screen;

in VertexData
{
    // output to fragment shader
    vec3 pass_point_color;
    vec3 pass_normal;
    vec2 pass_uv_coords;
}
VertexIn;

layout(location = 0) out vec4 out_color;


void main() {
  vec2 uv_coords = VertexIn.pass_uv_coords;

  if ( dot(uv_coords, uv_coords)> 1 ) {
    discard;
  }
  else {
    out_color = vec4(pow(VertexIn.pass_point_color, vec3(1.4,1.4,1.4)), 1.0);
  }
}
