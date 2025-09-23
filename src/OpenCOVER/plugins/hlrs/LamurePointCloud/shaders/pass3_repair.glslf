// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 440 core
 
layout(binding  = 0) uniform sampler2D in_color_texture;
layout(binding  = 1) uniform sampler2D depth_texture;

layout(location = 0) out vec4 out_color;
        
uniform vec3 background_color = vec3(0.0, 0.0, 0.0);

uniform vec2 win_size;

//for texture access
in vec2 pos;

void fetch_neighborhood_depth( inout float[8] in_out_neighborhood ) {
	in_out_neighborhood[0] = texture2D(depth_texture, (gl_FragCoord.xy + vec2(-1,+1) )/(win_size.xy) ).r; //upper left pixel
	in_out_neighborhood[1] = texture2D(depth_texture, (gl_FragCoord.xy + vec2(0,+1) )/(win_size.xy) ).r; //upper pixel
	in_out_neighborhood[2] = texture2D(depth_texture, (gl_FragCoord.xy + vec2(+1,+1) )/(win_size.xy) ).r; //upper right pixel
	in_out_neighborhood[3] = texture2D(depth_texture, (gl_FragCoord.xy + vec2(-1,0) )/(win_size.xy) ).r; //left pixel
	in_out_neighborhood[4] = texture2D(depth_texture, (gl_FragCoord.xy + vec2(+1,0) )/(win_size.xy) ).r; //right pixel
	in_out_neighborhood[5] = texture2D(depth_texture, (gl_FragCoord.xy + vec2(-1,-1) )/(win_size.xy) ).r; //lower left pixel
	in_out_neighborhood[6] = texture2D(depth_texture, (gl_FragCoord.xy + vec2(0,-1) )/(win_size.xy) ).r; //lower pixel
	in_out_neighborhood[7] = texture2D(depth_texture, (gl_FragCoord.xy + vec2(+1,-1) )/(win_size.xy) ).r; //lower right pixel
}

void main() {

  	float depthValue = texture2D(depth_texture, gl_FragCoord.xy/win_size.xy).r ;

	{

		if(depthValue != 0.0)
		  out_color = texture2D(in_color_texture, gl_FragCoord.xy/(win_size.xy));
		else
		{
	      

	      
	      float[8] neighborhood_depth;

	      fetch_neighborhood_depth(neighborhood_depth);

		
		  // neighborhood_depth neighbourhood indexing:
		  // 0 1 2
		  // 3   4
		  // 5 6 7

		  //pattern symbols:
		  //b = background pixel
		  //x = random 
		  //o = center pixel
		  
		 //rule 1:
		 //if all of the b-pixel are actually background pixel: pattern matches
		 //rule 2:
		 //if at least 1 pattern matches: don't fill

		  //test against pattern 0  
		  
                  //x b b    x 1 2
 		  //x o b    x   4
		  //x b b    x 6 7
		 
		 bool pattern0 = (neighborhood_depth[1] == 0.0) && (neighborhood_depth[2] == 0.0) && (neighborhood_depth[4] == 0.0) && (neighborhood_depth[6] == 0.0) && (neighborhood_depth[7] == 0.0) ;
		 
		 //test against pattern 1  
		  
                  //b b b    0 1 2
 		  //b o b    3   4
		  //x x x    x x x
	
		 bool pattern1 = (neighborhood_depth[0] == 0.0) && (neighborhood_depth[1] == 0.0) && (neighborhood_depth[2] == 0.0) && (neighborhood_depth[3] == 0.0) && (neighborhood_depth[4] == 0.0) ;

		 //test against pattern 2  
		  
                  //b b x    0 1 x
 		  //b o x    3   x
		  //b b x    5 6 x
	
		 bool pattern2 = (neighborhood_depth[0] == 0.0) && (neighborhood_depth[1] == 0.0) && (neighborhood_depth[3] == 0.0) && (neighborhood_depth[5] == 0.0) && (neighborhood_depth[6] == 0.0) ;

		 //test against pattern 3  
		  
                  //x x x    x x x
 		  //b o b    3   4
		  //b b b    5 6 7
	
		 bool pattern3 = (neighborhood_depth[3] == 0.0) && (neighborhood_depth[4] == 0.0) && (neighborhood_depth[5] == 0.0) && (neighborhood_depth[6] == 0.0) && (neighborhood_depth[7] == 0.0) ;

		 //test against pattern 4  
		  
                  //b b b    0 1 2
 		  //x o b    x   4
		  //x x b    x x 7
	
		 bool pattern4 = (neighborhood_depth[0] == 0.0) && (neighborhood_depth[1] == 0.0) && (neighborhood_depth[2] == 0.0) && (neighborhood_depth[4] == 0.0) && (neighborhood_depth[7] == 0.0) ;

		 //test against pattern 5  
		  
                  //b b b    0 1 2
 		  //b o x    3   x
		  //b x x    5 x x
	
		 bool pattern5 = (neighborhood_depth[0] == 0.0) && (neighborhood_depth[1] == 0.0) && (neighborhood_depth[2] == 0.0) && (neighborhood_depth[3] == 0.0) && (neighborhood_depth[5] == 0.0) ;

		 //test against pattern 6
		  
                  //b x x    0 x x
 		  //b o x    3   x
		  //b b b    5 6 7
	
		 bool pattern6 = (neighborhood_depth[0] == 0.0) && (neighborhood_depth[3] == 0.0) && (neighborhood_depth[5] == 0.0) && (neighborhood_depth[6] == 0.0) && (neighborhood_depth[7] == 0.0) ;

		 //test against pattern 7
		  
                  //x x b    x x 2
 		  //x o b    x   4
		  //b b b    5 6 7
	
		 bool pattern7 = (neighborhood_depth[2] == 0.0) && (neighborhood_depth[4] == 0.0) && (neighborhood_depth[5] == 0.0) && (neighborhood_depth[6] == 0.0) && (neighborhood_depth[7] == 0.0) ;


		//red means: is background and should be filled
		//yellow means: is background and should not be filled

 		  if( pattern0 || pattern1 || pattern2 || pattern3 || pattern4 || pattern5 || pattern6 || pattern7  ) 
		  {
		 	 out_color = vec4(background_color ,1.0);
		  }
		  else
		  {

		  	
			out_color = vec4(1.0,0.0,0.0,1.0);
			
			uint num_accumulated_colors = 0;
			vec3 accumulated_color = vec3(0.0, 0.0, 0.0);
			if(neighborhood_depth[0] != 0) {
				accumulated_color += texture2D(in_color_texture, (gl_FragCoord.xy + vec2(-1,+1) )/(win_size.xy) ).rgb;
				++num_accumulated_colors;
			}
			if(neighborhood_depth[1] != 0) {
				accumulated_color += texture2D(in_color_texture, (gl_FragCoord.xy + vec2(0,+1) )/(win_size.xy) ).rgb;
				++num_accumulated_colors;
			}

			if(neighborhood_depth[2] != 0) {
				accumulated_color += texture2D(in_color_texture, (gl_FragCoord.xy + vec2(+1,+1) )/(win_size.xy) ).rgb; //upper right pixel
				++num_accumulated_colors;
			}

			if(neighborhood_depth[3] != 0) {
				accumulated_color += texture2D(in_color_texture, (gl_FragCoord.xy + vec2(-1,0) )/(win_size.xy) ).rgb; //left pixel
				++num_accumulated_colors;
			}

			if(neighborhood_depth[4] != 0) {
				accumulated_color += texture2D(in_color_texture, (gl_FragCoord.xy + vec2(+1,0) )/(win_size.xy) ).rgb; //right pixel
				++num_accumulated_colors;
			}

			if(neighborhood_depth[5] != 0) {
				accumulated_color +=  texture2D(in_color_texture, (gl_FragCoord.xy + vec2(-1,-1) )/(win_size.xy) ).rgb; //lower left pixel
				++num_accumulated_colors;
			}

			if(neighborhood_depth[6] != 0) {
				accumulated_color += texture2D(in_color_texture, (gl_FragCoord.xy + vec2(0,-1) )/(win_size.xy) ).rgb; //lower pixel
				++num_accumulated_colors;
			}

			if(neighborhood_depth[7] != 0) {
				accumulated_color += texture2D(in_color_texture, (gl_FragCoord.xy + vec2(+1,-1) )/(win_size.xy) ).rgb; //lower right pixel
				++num_accumulated_colors;
			}

			if(num_accumulated_colors == 0)
				out_color = vec4(1.0, 0.0, 0.0, 1.0);
			else
				//out_color = vec4(1.0, 1.0, 1.0, 1.0);
				out_color = vec4(accumulated_color / num_accumulated_colors, 1.0);
		 	 out_color = vec4(0.0, 0.0 ,0.0 ,1.0);
			
				
		  }


			
			
		}





	}


 }
