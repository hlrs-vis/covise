// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 440 core

//#extension GL_NV_fragment_shader_interlock : require
#extension GL_NV_shader_atomic_float : require
//#extension GL_NV_gpu_shader5 : enable

in VertexData {
  vec2 pass_uv_coords;
  float es_depth;
  flat uint pass_further_global_surfel_id;
} VertexIn;

uniform float EPSILON;
uniform float near_plane;
uniform float far_plane;
uniform uint res_x;
uniform uint num_blended_frags;

layout(binding = 0, RGBA16UI) coherent uniform restrict uimageBuffer linked_list_buffer;
layout(binding = 1, R32UI) coherent uniform restrict uimage2D fragment_count_img;
layout(binding = 2, R32UI) coherent uniform restrict uimage2D min_es_distance_image;
//layout(binding = 4, offset = 0) uniform atomic_uint atomic_head_counter;

//#define EPSILON 0.001
#define MAX_INT_32 2147483647

uint uintify_float_depth(in float float_depth) {
  return uint( MAX_INT_32 - ( MAX_INT_32 * (float_depth - near_plane) / (far_plane - near_plane) ) );
}

float floatify_uint_depth(in uint uint_depth) {
  return float( ( float((MAX_INT_32 - uint_depth)) / float(MAX_INT_32) ) * (far_plane-near_plane) + near_plane);
}

void main()
{
  
  if( dot(VertexIn.pass_uv_coords, VertexIn.pass_uv_coords) > 1 )
    discard;

  float own_depth = VertexIn.es_depth;

  uint uintified_own_depth = uintify_float_depth(own_depth);

  //imageAtomicMax(min_es_distance_image, ivec2(gl_FragCoord.xy), 10000.0);
  uint current_min_depth = imageAtomicMax(min_es_distance_image, ivec2(gl_FragCoord.xy), uintified_own_depth ).x;

  float floatified_min_depth = floatify_uint_depth(current_min_depth);
  if(own_depth > floatified_min_depth + EPSILON)
    discard;


  uint global_surfel_id = VertexIn.pass_further_global_surfel_id;

  uint weight_index = uint(floor(length(VertexIn.pass_uv_coords) * 20.0 ));
  uint undecomposed_converted_depth = floatBitsToUint(VertexIn.es_depth);
  uvec4 insert_into_list = uvec4(undecomposed_converted_depth & 0x0000FFFF, 
                                (undecomposed_converted_depth >> 16) & 0x0000FFFF,
                                global_surfel_id & 0x0000FFFF, 
                                ( (global_surfel_id >> 16) & 0x000007FF ) |  ( (weight_index ) << 11 ) );


  uint pixel_chunk_start = (uint(gl_FragCoord.x)  +  uint(gl_FragCoord.y) * res_x) * num_blended_frags;

  uint currently_written_fragments = imageAtomicAdd(fragment_count_img, ivec2(gl_FragCoord.xy), uint(1) );

  bool already_done = false;

  uint write_offset = 0;

  uint local_offset = pixel_chunk_start;
  if(currently_written_fragments < num_blended_frags) {

    write_offset = currently_written_fragments;
    uint new_idx = pixel_chunk_start + write_offset;

    imageStore(linked_list_buffer, int(new_idx), insert_into_list);
  } else {
    discard;
  }



}

