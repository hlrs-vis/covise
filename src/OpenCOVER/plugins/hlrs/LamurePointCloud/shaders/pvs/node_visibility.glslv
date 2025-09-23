// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

uniform mat4 inv_mv_matrix;
uniform float point_size_factor;
uniform float model_radius_scale;

layout(location = 0) in vec3 in_position;
layout(location = 5) in float in_radius;
layout(location = 6) in vec3 in_normal;


out VertexData {
  vec3 pass_ms_u;
  vec3 pass_ms_v;
  vec3 pass_normal;
} VertexOut;

INCLUDE ../common/compute_tangent_vectors.glsl

void main() {

  vec3 tangent = vec3(0.0);
  vec3 bitangent = vec3(0.0);

  compute_tangent_vectors(in_normal, in_radius, tangent, bitangent);
  vec3 normal = normalize((inv_mv_matrix * vec4(in_normal, 0.0)).xyz);

  // passed attributes: vertex shader -> geometry shader
  VertexOut.pass_ms_u = tangent;
  VertexOut.pass_ms_v = bitangent;
  VertexOut.pass_normal = normal;
  gl_Position = vec4(in_position, 1.0);
}