// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

uniform mat4 model_to_screen_matrix;
uniform mat4 inv_mv_matrix;
uniform float model_radius_scale;
uniform float point_size_factor;

uniform float radius_sphere;
uniform vec3 position_sphere;
uniform bool render_normals;
uniform bool state_lense;
uniform int mode_prov_data;
uniform float heatmap_min;
uniform float heatmap_max;
uniform vec3 heatmap_min_color;
uniform vec3 heatmap_max_color;

layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 4) in float empty;
layout(location = 5) in float in_radius;
layout(location = 6) in vec3 in_normal;


out VertexData {
  //output to geometry shader
  vec3 pass_ms_u;
  vec3 pass_ms_v;

  vec3 pass_point_color;
  vec3 pass_normal;
} VertexOut;

INCLUDE common/compute_tangent_vectors.glsl
INCLUDE common/provenance_visualization_functions.glsl

void main()
{
  // precalculate tangent vectors to establish the surfel shape
  vec3 tangent   = vec3(0.0);
  vec3 bitangent = vec3(0.0);
  compute_tangent_vectors(in_normal, in_radius, tangent, bitangent);

  // finalize normal and tangents
  vec3 normal = normalize((inv_mv_matrix * vec4(in_normal, 0.0)).xyz );

  // finalize color with provenance overlay
  vec3 in_out_color = vec3(in_r, in_g, in_b);
  resolve_provenance_coloring(in_position, normal, tangent, bitangent, in_radius, in_out_color);
  
  // passed attributes: vertex shader -> geometry shader
  VertexOut.pass_ms_u = tangent;
  VertexOut.pass_ms_v = bitangent;
  VertexOut.pass_normal = normal;
  gl_Position = vec4(in_position, 1.0);
  VertexOut.pass_point_color = in_out_color;

}
