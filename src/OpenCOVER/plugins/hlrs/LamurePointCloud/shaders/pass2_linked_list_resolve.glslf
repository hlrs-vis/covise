// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 440 core

//#extension GL_NV_gpu_shader5 : enable

struct PointAttributeData {
  float position[3];
  uint rgb32;
  float radius;
  float normal[3];
};

layout(std430, binding=0) buffer point_attrib_ssbo {
  PointAttributeData attributes[];
};

const float gaussian_weights[20] = float[](
  1.000000, 1.000000, 0.968627,  0.917647,  0.870588,  0.788235,
  0.690196,  0.619608,  0.513725,  0.458824,  0.356863,
  0.278431,  0.227451, 0.164706, 0.152941, 0.125490, 0.109804, 0.098039,
  0.074510, 0.062745
);

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_normal;

layout(binding = 0, RGBA16UI) readonly coherent uniform restrict uimageBuffer linked_list_buffer;
layout(binding = 1, R32UI) readonly coherent uniform restrict uimage2D fragment_count_img;
layout(binding = 2, R32UI) coherent uniform restrict uimage2D min_es_distance_image;

//for texture access
in vec2 pos;
uniform uint res_x;
uniform uint num_blended_frags;
uniform float EPSILON;
uniform float near_plane;
uniform float far_plane;

#define attr_tex_width  4096
//#define EPSILON 0.001
#define MAX_INT_32 2147483647
#define NUM_BLEND_FRAGS 18


uint uintify_float_depth(in float float_depth) {
  return uint( MAX_INT_32 - ( MAX_INT_32 * (float_depth - near_plane) / (far_plane - near_plane) ) );
}

float floatify_uint_depth(in uint uint_depth) {
  return float( ( float((MAX_INT_32 - uint_depth)) / float(MAX_INT_32) ) * (far_plane-near_plane) + near_plane);
}

float gathered_depth[NUM_BLEND_FRAGS];
uint surfel_idx_and_weight[NUM_BLEND_FRAGS];


void bubbleSort()
{
  int n = 0;
  do{
    int new_n = 1;
    for ( int i=0; i<(NUM_BLEND_FRAGS-1); ++i){
      if (gathered_depth[i] > gathered_depth[i+1]){
        float tmp_float = gathered_depth[i];
        gathered_depth[i] = gathered_depth[i+1];
        gathered_depth[i+1] = tmp_float;

        uint tmp_uint = surfel_idx_and_weight[i];
        surfel_idx_and_weight[i] = surfel_idx_and_weight[i+1];
        surfel_idx_and_weight[i+1] = tmp_uint;

        new_n = i+1;
      } // ende if
    } // ende for
    n = new_n;
  } while (n > 1);
}

void main()	
{

  	vec4 looked_up_color = vec4(0.0, 0.0, 0.0, 0.0);
  	vec3 looked_up_normal = vec3(0.0, 0.0, 0.0);

  	uint frag_count = imageLoad(fragment_count_img, ivec2(gl_FragCoord.xy)).x;
  	uint limit = min(frag_count , num_blended_frags);

  	float nearest_es_depth = floatify_uint_depth(imageLoad(min_es_distance_image, ivec2(gl_FragCoord.xy)).x );

  	uint premultiplied_offset = (uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * res_x)*num_blended_frags;


	//NUM_BLENDED_FRAGS
  	for(uint frag_offset = 0; frag_offset < limit; ++frag_offset) {

  		if(limit > frag_offset) {
		  	uint new_idx = premultiplied_offset + frag_offset;

			uvec4 depth_LMSB_ID_LMSB = imageLoad(linked_list_buffer, int(new_idx) );

			uvec2 retrieved_attributes = uvec2(
												(depth_LMSB_ID_LMSB.x & 0xFFFF) |  ((depth_LMSB_ID_LMSB.y & 0xFFFF) << 16), // former total_depth_as_int
												//(depth_LMSB_ID_LMSB.z & 0xFFFF) |  ((depth_LMSB_ID_LMSB.w & 0x07FF) << 16)  // former global_surfel_idx
												(depth_LMSB_ID_LMSB.z & 0xFFFF) |  ((depth_LMSB_ID_LMSB.w & 0xFFFF) << 16)  // former global_surfel_idx
							);
			float es_depth = uintBitsToFloat(retrieved_attributes.x);

			gathered_depth[frag_offset] = es_depth;
			surfel_idx_and_weight[frag_offset] = retrieved_attributes.y;

		}
  	}

  	for(uint frag_offset = limit; frag_offset < NUM_BLEND_FRAGS; ++frag_offset) {
  		gathered_depth[frag_offset] = 100000.0;
  	}

  	bubbleSort();

		uint surfel_idx = surfel_idx_and_weight[0] & 0x07FFFFFF;
		uint weight_index = ((surfel_idx_and_weight[0]>> 16) & 0x0000F800) >> 11;

  	float front_most_radius = attributes[surfel_idx].radius;

  	for(int i = 0; i < NUM_BLEND_FRAGS; ++i) {

  		float es_depth = gathered_depth[i];
  		
  		if( es_depth > nearest_es_depth + 0.001 + 0.01*(front_most_radius) )
			break;

			//nearest_es_depth = es_depth;

		uint surfel_idx = surfel_idx_and_weight[i] & 0x07FFFFFF;
		uint weight_index = ((surfel_idx_and_weight[i]>> 16) & 0x0000F800) >> 11;

	  float current_weight = gaussian_weights[weight_index];


		uint SSBO_COLOR = attributes[surfel_idx].rgb32;
		float[3] SSBO_NORMAL = attributes[surfel_idx].normal;

		looked_up_normal += current_weight * vec3(SSBO_NORMAL[0],
																							SSBO_NORMAL[1],
																							SSBO_NORMAL[2]);

		looked_up_color += vec4( current_weight * vec3(float((attributes[surfel_idx].rgb32>>0) & 255),
																									 float((attributes[surfel_idx].rgb32>>8) & 255),
																									 float((attributes[surfel_idx].rgb32>>16) & 255)) / 255.0, current_weight );
  	}

/*

*/ //USE THIS LATER


  	looked_up_normal /= looked_up_color.w;
	looked_up_color /= looked_up_color.w;

	looked_up_normal.z = pow( 1.0 - dot(looked_up_normal.xy, looked_up_normal.xy), 0.5 );
/*
	uint atomic_value = imageLoad(min_es_distance_image, ivec2(gl_FragCoord.xy)).x ;

	if(atomic_value > 0)
		out_color = vec3(0.0, 1.0, 0.0);
	else
		out_color = vec3(1.0, 0.0, 0.0);
*/

	if(frag_count > num_blended_frags) {
		//out_color = vec3(1.0, 1.0, 1.0) ;
		imageStore(min_es_distance_image, ivec2(gl_FragCoord.xy), uvec4(0, 0, 0, 0) ); 
	} else {
		out_color = vec3(pow( looked_up_color.xyz, vec3(1.4,1.4,1.4) )) ;	
	}


	//out_color = vec3(front_most_radius*10,0.0, 0.0);
	out_normal = looked_up_normal;

 }
