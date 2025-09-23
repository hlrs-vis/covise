// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

INCLUDE common/attribute_dequantization_header.glsl

uniform mat4 model_to_screen_matrix;
uniform mat4 inv_mv_matrix;
uniform float model_radius_scale;
uniform float point_size_factor;


out VertexData {
  //output to geometry shader
  vec3 pass_ms_u;
  vec3 pass_ms_v;

  vec3 pass_point_color;
  vec3 pass_normal;
} VertexOut;

INCLUDE common/attribute_dequantization_functions.glsl
INCLUDE common/compute_tangent_vectors.glsl
INCLUDE common/provenance_visualization_functions.glsl

void main() {
  // the "in" prefix is kept to be consistent with the interface of the uncompressed shaders.
  // conceptually, the decompressed attributes are the input for the next stages.
  vec3  in_position = vec3(0.0);
  vec3  in_normal   = vec3(0.0);
  float in_radius  = 0.0;
  vec3  in_rgb = vec3(0.0);

  dequantize_surfel_attributes_full(in_qz_pos_xy_16_16, in_qz_pos_z_normal_enum_16_16, in_rgb_777_and_radius_11, //compressed v-attributes
                                    in_position, in_normal, in_radius, in_rgb); //decompressed v-attributes

  // precalculate tangent vectors to establish the surfel shape
  vec3 tangent   = vec3(0.0);
  vec3 bitangent = vec3(0.0);
  compute_tangent_vectors(in_normal, in_radius, tangent, bitangent);

  // finalize normal and tangents
  vec3 normal = normalize((inv_mv_matrix * vec4(in_normal, 0.0)).xyz );

  // finalize color with provenance overlay
  vec3 in_out_color = vec3(in_rgb);
  resolve_provenance_coloring(in_position, normal, tangent, bitangent, in_radius, in_out_color);//, in_normal, tangent, bitangent, in_radius, in_out_color);
  
  // passed attributes: vertex shader -> geometry shader
  VertexOut.pass_normal = normal;
  VertexOut.pass_ms_u = tangent;
  VertexOut.pass_ms_v = bitangent;
  gl_Position = vec4(in_position, 1.0);
  VertexOut.pass_point_color = in_out_color;
}
