// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#extension GL_ARB_shader_storage_buffer_object : require

layout(location = 0) in uint in_qz_pos_xy_16_16;
layout(location = 1) in uint in_qz_pos_z_normal_enum_16_16;
layout(location = 2) in uint in_rgb_777_and_radius_11;

uniform int num_primitives_per_node; 

struct bvh_auxiliary {
  vec4 bb_and_rad_min;
  vec4 bb_and_rad_max;
};

layout(std430, binding = 1) coherent readonly buffer bvh_auxiliary_struct {
     bvh_auxiliary data_bvh[];
};
