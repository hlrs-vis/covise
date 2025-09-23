// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core
 
layout(binding  = 0) uniform sampler2D in_color_texture;
layout(binding  = 1) uniform sampler2D depth_texture;

layout(location = 0) out vec4 out_color;
        
uniform vec3 background_color = vec3(0.0, 0.0, 0.0);

const vec3 luminocity_blend_values = vec3(0.2126, 0.7152, 0.0722);

const ivec2 neighborhood_access_array[8] = { {-1, +1}, {0,+1}, {+1,+1}, {-1,0}, {+1,0}, {-1,-1}, {0,-1}, {+1,-1} };
//for texture access
in vec2 pos;

void fetch_neighborhood_depth( inout float[8] in_out_neighborhood ) {

  for(int neighbor_idx = 0; neighbor_idx < 8; ++neighbor_idx) {
	in_out_neighborhood[neighbor_idx] = texelFetch(depth_texture, ivec2(gl_FragCoord.xy + neighborhood_access_array[neighbor_idx]), 0).r;
  }
}

void main() {

  	float center_depth_value = texelFetch(depth_texture, ivec2(gl_FragCoord.xy), 0).r ;

	{
		if(center_depth_value != 1.0f)
		  out_color = texelFetch(in_color_texture, ivec2(gl_FragCoord.xy), 0);
		else
		{
	      
	      float[8] neighborhood_depth;

	      fetch_neighborhood_depth(neighborhood_depth);

		
		  // neighborhood_depth neighbourhood indexing:
		  // 0 1 2
		  // 3 c 4
		  // 5 6 7

		  //pattern symbols:
		  //b = background pixel
		  //x = random 
		  //o = center pixel
		  
		 //rule 1:
		 //if all of the b-pixel are actually background pixel: pattern matches
		 //rule 2:
		 //if at least 1 pattern matches: don't fill



		  bool[8] is_neighborhood_pixel_background;

		  for(int neighborhood_idx = 0; neighborhood_idx < 8; ++neighborhood_idx) {
		  	is_neighborhood_pixel_background[neighborhood_idx] = 1.0 == neighborhood_depth[neighborhood_idx];
		  }

		  //test against pattern 0  
          //x b b    x 1 2
 		  //x o b    x   4
		  //x b b    x 6 7
		 bool pattern0 = (is_neighborhood_pixel_background[1]) && (is_neighborhood_pixel_background[2]) && (is_neighborhood_pixel_background[4]) && (is_neighborhood_pixel_background[6]) && (is_neighborhood_pixel_background[7]);
		 
		 //test against pattern 1   
          //b b b    0 1 2
 		  //b o b    3   4
		  //x x x    x x x
		 bool pattern1 = (is_neighborhood_pixel_background[0]) && (is_neighborhood_pixel_background[1]) && (is_neighborhood_pixel_background[2]) && (is_neighborhood_pixel_background[3]) && (is_neighborhood_pixel_background[4]);
		 
		 //test against pattern 2   
          //b b x    0 1 x
 		  //b o x    3   x
		  //b b x    5 6 x
		 bool pattern2 = (is_neighborhood_pixel_background[0]) && (is_neighborhood_pixel_background[1]) && (is_neighborhood_pixel_background[3]) && (is_neighborhood_pixel_background[5]) && (is_neighborhood_pixel_background[6]);
		 
		 //test against pattern 3   
          //x x x    x x x
 		  //b o b    3   4
		  //b b b    5 6 7
		 bool pattern3 = (is_neighborhood_pixel_background[3]) && (is_neighborhood_pixel_background[4]) && (is_neighborhood_pixel_background[5]) && (is_neighborhood_pixel_background[6]) && (is_neighborhood_pixel_background[7]);
		 
		 //test against pattern 4  
		  
          //b b b    0 1 2
 		  //x o b    x   4
		  //x x b    x x 7
		 bool pattern4 = (is_neighborhood_pixel_background[0]) && (is_neighborhood_pixel_background[1]) && (is_neighborhood_pixel_background[2]) && (is_neighborhood_pixel_background[4]) && (is_neighborhood_pixel_background[7]);
		 
		 //test against pattern 5  
          //b b b    0 1 2
 		  //b o x    3   x
		  //b x x    5 x x
		 bool pattern5 = (is_neighborhood_pixel_background[0]) && (is_neighborhood_pixel_background[1]) && (is_neighborhood_pixel_background[2]) && (is_neighborhood_pixel_background[3]) && (is_neighborhood_pixel_background[5]);
		 
		 //test against pattern 6 
          //b x x    0 x x
 		  //b o x    3   x
		  //b b b    5 6 7
		 bool pattern6 = (is_neighborhood_pixel_background[0]) && (is_neighborhood_pixel_background[3]) && (is_neighborhood_pixel_background[5]) && (is_neighborhood_pixel_background[6]) && (is_neighborhood_pixel_background[7]);
		 
		 //test against pattern 7 
          //x x b    x x 2
 		  //x o b    x   4
		  //b b b    5 6 7
		 bool pattern7 = (is_neighborhood_pixel_background[2]) && (is_neighborhood_pixel_background[4]) && (is_neighborhood_pixel_background[5]) && (is_neighborhood_pixel_background[6]) && (is_neighborhood_pixel_background[7]);

		//red means: is background and should be filled
		//yellow means: is background and should not be filled

 		  if( pattern0 || pattern1 || pattern2 || pattern3 || pattern4 || pattern5 || pattern6 || pattern7  ) 
		  {
		 	 out_color = vec4(0.0, 0.0, 0.0, 1.0);
		 	 //put bg color here
		 	 out_color = vec4(background_color.rgb ,1.0);
		  }
		  else
		  {
			out_color = vec4(1.0, 0.0, 0.0 ,1.0);
			

			// re-fill the neighborhood_depth array with luminocity values of the neighborhood_depth area to reuse it 
			// for finding the median luminosity color

			for(int k = 0; k < 8; ++k) {
				vec3 tempCol = texelFetch(in_color_texture, ivec2(gl_FragCoord.xy + neighborhood_access_array[k]), 0).rgb; //upper left pixel
				neighborhood_depth[k] = dot(luminocity_blend_values, tempCol);
			}

			//find the median element with index 4
			for(int i = 0; i < 8; ++i) {

			int sum_smaller_elements = 0;
			int sum_equal_elements = 0;

				for(int k = 0; k < 8; ++k) {
					if(i != k) {
						if(neighborhood_depth[i] >= neighborhood_depth[k]) { //our current element was smaller, so we don't have to do anything
							if(neighborhood_depth[i] > neighborhood_depth[k]) {
							  sum_smaller_elements += 1;
							} else {
							  sum_equal_elements += 1;
							}
						}
					}
				}

				if( (sum_smaller_elements + sum_equal_elements >= 3) ) {

					vec4 tempC = texelFetch(in_color_texture, ivec2(gl_FragCoord.xy + neighborhood_access_array[i]), 0);
					
					if( (tempC.rgb == vec3(0.0,0.0,0.0) ) && i != 7 ) {
						continue;
					} else {
						out_color = tempC;
					}
					
					break;
				}
			}

		  }



		}





	}

 }
