// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core


out VertexData {
  vec3 pass_ms_u;
  vec3 pass_ms_v;
  vec3 pass_normal;
} VertexOut;

uniform mat4 mvp_matrix;
uniform mat4 model_matrix;
uniform mat4 model_view_matrix;
uniform mat4 inv_mv_matrix;
uniform mat4 model_to_screen_matrix;

uniform float near_plane;

uniform bool face_eye;
uniform vec3 eye;
uniform float max_radius;

uniform float point_size_factor;
uniform float model_radius_scale;


layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 4) in float empty;
layout(location = 5) in float in_radius;
layout(location = 6) in vec3 in_normal;


INCLUDE ../common/compute_tangent_vectors.glsl

void main() {
  float radius = in_radius;
  if (radius > max_radius) {
    radius = max_radius;
  }

  vec3 normal = in_normal;
  if (face_eye) {
    normal = normalize(eye-(model_matrix*vec4(in_position, 1.0)).xyz);
  }
 
  // precalculate tangent vectors to establish the surfel shape
  vec3 tangent   = vec3(0.0);
  vec3 bitangent = vec3(0.0);
  compute_tangent_vectors(normal, radius, tangent, bitangent);

  VertexOut.pass_ms_u = tangent;
  VertexOut.pass_ms_v = bitangent;
  VertexOut.pass_normal = normal;
  gl_Position = vec4(in_position, 1.0);
}