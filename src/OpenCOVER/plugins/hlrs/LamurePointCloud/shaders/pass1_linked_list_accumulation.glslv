// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 440 core

uniform float point_size_factor;
uniform float model_radius_scale;


layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 4) in float empty;
layout(location = 5) in float in_radius;
layout(location = 6) in vec3 in_normal;

layout(binding = 0, RGBA16UI) coherent uniform restrict uimageBuffer linked_list_buffer;

out VertexData {
  vec3 pass_ms_u;
  vec3 pass_ms_v;
  flat uint pass_global_surfel_id;
} VertexOut;

INCLUDE common/compute_tangent_vectors.glsl

void main() {
  // precalculate tangent vectors to establish the surfel shape
  vec3 tangent = vec3(0.0);
  vec3 bitangent = vec3(0.0);

  compute_tangent_vectors(in_normal, in_radius, tangent, bitangent);

  // passed attributes: vertex shader -> geometry shader
  VertexOut.pass_ms_u = tangent;
  VertexOut.pass_ms_v = bitangent;
  gl_Position = vec4(in_position, 1.0);

  uint global_surfel_idx = gl_VertexID;

  VertexOut.pass_global_surfel_id = global_surfel_idx;
}