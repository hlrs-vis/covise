// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

const int num_points_u = 104;
const int num_points_v = 105;
const int total_num_points_per_face = 104 * 105;
const int face_divisor = 2;

const ivec3 rgb_777_shift_vec   = ivec3(25, 18, 11);
const ivec3 rgb_777_mask_vec    = ivec3(0x7F, 0x7F, 0x7F);
const ivec3 rgb_777_quant_steps = ivec3(2, 2, 2); 

  void dequantize_position_16_16_16(in uint qz_pos_xy, in uint qz_pos_z_normal_enum, in vec3 bb_min_pos, in vec3 bb_max_pos,
                                    out vec3 out_position) {

  uint in_qz_pos_y = ( qz_pos_xy >> 16 ) & 0xFFFF;
  uint in_qz_pos_x = qz_pos_xy & 0xFFFF;
  uint in_qz_pos_z = ( qz_pos_z_normal_enum ) & 0xFFFF;

  // dequantize position
  out_position = bb_min_pos + ivec3(in_qz_pos_x, in_qz_pos_y, in_qz_pos_z) * ( (bb_max_pos - bb_min_pos)/65535.0 );
}

void dequantize_radius_11(in uint rgb777_and_radius_11, in float rad_min, in float rad_max,
                          out float out_radius) {

  float radius_to_assign = 0.0;

  uint radius_quantization_index = (rgb777_and_radius_11 & 0x7FF);

  if( 2047 != radius_quantization_index) {
    radius_to_assign = rad_min + (radius_quantization_index) * ( (rad_max  - rad_min) / 2047.0);
  }

  out_radius = radius_to_assign;
}

void dequantize_normal_16( in uint qz_pos_z_normal_enum,
                           out vec3 out_normal) {

  // dequantized normal
  int qz_normal_enumerator = int (( qz_pos_z_normal_enum >> 16 ) & 0xFFFF);
  int face_id = qz_normal_enumerator / (total_num_points_per_face);
  int is_main_axis_negative = ((face_id % 2) == 1 ) ? -1 : 1;
  qz_normal_enumerator -= face_id * (total_num_points_per_face);
  int v_component = qz_normal_enumerator / num_points_u;
  int u_component = qz_normal_enumerator % num_points_u;

  int main_axis = face_id / face_divisor;

  float first_component = (u_component / float(num_points_u) ) * 2.0 - 1.0;
  float second_component = (v_component / float(num_points_v) ) * 2.0 - 1.0;

  vec3 unswizzled_components = vec3(first_component, 
                                    second_component,
                                    is_main_axis_negative * sqrt(-(first_component*first_component) - (second_component*second_component) + 1) );

  vec3 normal_to_assign = unswizzled_components.zxy;

  if(1 == main_axis) {
    normal_to_assign.xyz = unswizzled_components.yzx;
  } else if(2 == main_axis) {
    normal_to_assign.xyz = unswizzled_components.xyz;    
  }

  out_normal = normal_to_assign;
}

void dequantize_color_777(in uint rgb_777_and_radius_11, out vec3 out_rgb) {
  ivec3 rgb_multiplier = (ivec3(rgb_777_and_radius_11) >> rgb_777_shift_vec) & rgb_777_mask_vec;
  out_rgb = rgb_multiplier * rgb_777_quant_steps / 255.0;
}

void dequantize_surfel_attributes_without_color(in uint qz_pos_xy, in uint qz_pos_z_normal_enum, in uint rgb_777_and_radius_11,
                                                out vec3 out_position, out vec3 out_normal, out float out_radius) {

  // vertexID relative to complete VBO / surfels per node = implicit index into BVH-SSBO
  int ssbo_node_id = gl_VertexID / num_primitives_per_node;
  // retrieve position context and radius context from cut-bvh ssbo
  bvh_auxiliary node_info = data_bvh[ssbo_node_id];

  // dequantization subfunctions
  dequantize_position_16_16_16(qz_pos_xy, qz_pos_z_normal_enum, node_info.bb_and_rad_min.xyz, node_info.bb_and_rad_max.xyz, out_position);
  dequantize_radius_11(rgb_777_and_radius_11, node_info.bb_and_rad_min.a, node_info.bb_and_rad_max.a, out_radius);
  dequantize_normal_16(qz_pos_z_normal_enum, out_normal);


}
void dequantize_surfel_attributes_full(in uint qz_pos_xy, in uint qz_pos_z_normal_enum, in uint rgb_777_and_radius_11,
                                       out vec3 out_position, out vec3 out_normal, out float out_radius, out vec3 out_rgb) {
  dequantize_surfel_attributes_without_color(qz_pos_xy, qz_pos_z_normal_enum, rgb_777_and_radius_11, out_position, out_normal, out_radius);
  dequantize_color_777(rgb_777_and_radius_11, out_rgb);
}