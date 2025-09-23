// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

out VertexData {
	vec3 nor;
  float rad;
	float mv_vertex_depth;
} VertexOut;

uniform mat4 mvp_matrix;
uniform mat4 model_view_matrix;
uniform mat4 inv_mv_matrix;

uniform float height_divided_by_top_minus_bottom;
uniform float near_plane;

uniform float point_size_factor;
uniform float model_radius_scale;

INCLUDE ../common/attribute_dequantization_header.glsl

layout(location = 3) in float prov1;
layout(location = 4) in float prov2;
layout(location = 5) in float prov3;
layout(location = 6) in float prov4;
layout(location = 7) in float prov5;
layout(location = 8) in float prov6;

INCLUDE ../common/attribute_dequantization_functions.glsl

void main()
{
  // the "in" prefix is kept to be consistent with the interface of the uncompressed shaders.
  // conceptually, the decompressed attributes are the input for the next stages.
  vec3  in_position = vec3(0.0);
  vec3  in_normal   = vec3(0.0);
  float in_radius  = 0.0;
  vec3  in_rgb = vec3(0.0);

  dequantize_surfel_attributes_full(
    in_qz_pos_xy_16_16, 
    in_qz_pos_z_normal_enum_16_16, 
    in_rgb_777_and_radius_11, //compressed v-attributes
    in_position, in_normal, in_radius, in_rgb); //decompressed v-attributes

  if(in_radius == 0.0) {
    gl_Position = vec4(2.0,2.0,2.0,1.0);
  }
  else {
    float scaled_radius = model_radius_scale * in_radius * point_size_factor;
    vec4 normal = inv_mv_matrix * vec4(in_normal,0.0f);
    vec4 pos_es = model_view_matrix * vec4(in_position, 1.0f);

    float ps = 3.0f*(scaled_radius) * (near_plane/-pos_es.z) * height_divided_by_top_minus_bottom;
    gl_Position = mvp_matrix * vec4(in_position, 1.0);

    VertexOut.nor = normal.xyz;
    gl_PointSize = ps;
    VertexOut.rad = (scaled_radius);
    VertexOut.mv_vertex_depth = pos_es.z;

   }

}

