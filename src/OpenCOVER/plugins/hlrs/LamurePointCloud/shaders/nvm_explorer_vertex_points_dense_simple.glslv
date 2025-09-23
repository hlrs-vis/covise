// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

uniform mat4 model_to_screen_matrix;
uniform mat4 inv_mv_matrix;
uniform mat4 mvp_matrix;
uniform float model_radius_scale;
uniform float point_size_factor;
// uniform int render_provenance;
uniform float average_radius;
uniform float accuracy;

layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 4) in float empty;
layout(location = 5) in float in_radius;
layout(location = 6) in vec3 in_normal;

out vec4 passed_color;
// out VertexData {
//   //output to geometry shader
//   vec3 pass_ms_u;
//   vec3 pass_ms_v;

//   vec3 pass_point_color;
//   vec3 pass_normal;
// } VertexOut;

// --------------------------
float Gamma        = 0.80;
float IntensityMax = 255.0;
	
float round(float d){
  return floor(d + 0.5);
}
	
float Adjust(float Color, float Factor){
  if (Color == 0.0){
    return 0.0;
  }
  else{
    float res = round(IntensityMax * pow(Color * Factor, Gamma));
    return min(255.0, max(0.0, res));
  }
}

vec3 WavelengthToRGB(float Wavelength){
  float Blue;
  float factor;
  float Green;
  float Red;
  if(380.0 <= Wavelength && Wavelength <= 440.0){
    Red   = -(Wavelength - 440.0) / (440.0 - 380.0);
    Green = 0.0;
    Blue  = 1.0;
  }
  else if(440.0 < Wavelength && Wavelength <= 490.0){
    Red   = 0.0;
    Green = (Wavelength - 440.0) / (490.0 - 440.0);
    Blue  = 1.0;
  }
  else if(490.0 < Wavelength && Wavelength <= 510.0){
    Red   = 0.0;
    Green = 1.0;
    Blue  = -(Wavelength - 510.0) / (510.0 - 490.0);
  }
  else if(510.0 < Wavelength && Wavelength <= 580.0){
    Red   = (Wavelength - 510.0) / (580.0 - 510.0);
    Green = 1.0;
    Blue  = 0.0;
  }
  else if(580.0 < Wavelength && Wavelength <= 645.0){		
    Red   = 1.0;
    Green = -(Wavelength - 645.0) / (645.0 - 580.0);
    Blue  = 0.0;
  }
  else if(645.0 < Wavelength && Wavelength <= 780.0){
    Red   = 1.0;
    Green = 0.0;
    Blue  = 0.0;
  }
  else{
    Red   = 0.0;
    Green = 0.0;
    Blue  = 0.0;
  }
  
  
  if(380.0 <= Wavelength && Wavelength <= 420.0){
    factor = 0.3 + 0.7*(Wavelength - 380.0) / (420.0 - 380.0);
  }
  else if(420.0 < Wavelength && Wavelength <= 701.0){
    factor = 1.0;
  }
  else if(701.0 < Wavelength && Wavelength <= 780.0){
    factor = 0.3 + 0.7*(780.0 - Wavelength) / (780.0 - 701.0);
  }
  else{
    factor = 0.0;
  }
  float R = Adjust(Red,   factor);
  float G = Adjust(Green, factor);
  float B = Adjust(Blue,  factor);
  return vec3(R/255.0,G/255.0,B/255.0);
}
	
	
	
	
float GetWaveLengthFromDataPoint(float Value, float MinValue, float MaxValue){
  float MinVisibleWavelength = 380.0;//350.0;
  float MaxVisibleWavelength = 780.0;//650.0;
  //Convert data value in the range of MinValues..MaxValues to the 
  //range 350..780
  return (Value - MinValue) / (MaxValue-MinValue) * (MaxVisibleWavelength - MinVisibleWavelength) + MinVisibleWavelength;
}	
	
	
vec3 DataPointToColor(float Value, float MinValue, float MaxValue){
  float Wavelength = GetWaveLengthFromDataPoint(Value, MinValue, MaxValue);
  return WavelengthToRGB(Wavelength);	  
}
// ------------------

void main()
{
  // vec3 ms_n = normalize(in_normal.xyz);
  // vec3 ms_u;

  //**compute tangent vectors**//
  // if(ms_n.z != 0.0) {
  //   ms_u = vec3( 1, 1, (-ms_n.x -ms_n.y)/ms_n.z);
  // } else if (ms_n.y != 0.0) {
  //   ms_u = vec3( 1, (-ms_n.x -ms_n.z)/ms_n.y, 1);
  // } else {
  //   ms_u = vec3( (-ms_n.y -ms_n.z)/ms_n.x, 1, 1);
  // }

  // // **assign tangent vectors**//
  // VertexOut.pass_ms_u = normalize(ms_u) * point_size_factor * model_radius_scale * in_radius;
  // VertexOut.pass_ms_v = normalize(cross(ms_n, ms_u)) * point_size_factor * model_radius_scale * in_radius;

  // VertexOut.pass_normal = normalize((inv_mv_matrix * vec4(in_normal, 0.0)).xyz );

  // gl_Position = vec4(in_position, 1.0);
    gl_PointSize = 1.0;

  passed_color = vec4(in_r, in_g, in_b, empty);

  gl_Position = mvp_matrix * vec4(in_position, 1.0);

  // VertexOut.pass_point_color = vec3(in_r, in_g, in_b);

  // int render_provenance = 2;
  // switch(render_provenance){
  //    case 1:
  //       {	
	 //    float ideal_screen_surfel_size = 2.0; // error threshold
	 //    float min_screen_surfel_size = 0.0; // error threshold
	 //    float max_screen_surfel_size = 10.0; // error threshold
  //       vec4 surfel_pos_screen = model_to_screen_matrix * vec4(in_position ,1.0);
  //       surfel_pos_screen /= surfel_pos_screen.w;
  //       vec4 border_pos_screen_u = model_to_screen_matrix * vec4(in_position + VertexOut.pass_ms_u,1.0);
  //       border_pos_screen_u /= border_pos_screen_u.w;
  //       vec4 border_pos_screen_v = model_to_screen_matrix * vec4(in_position + VertexOut.pass_ms_v,1.0);
  //       border_pos_screen_v /= border_pos_screen_v.w;
  //       float screen_surfel_size = max(length(surfel_pos_screen.xy - border_pos_screen_u.xy), length(surfel_pos_screen.xy - border_pos_screen_v.xy));
  //       screen_surfel_size = clamp(screen_surfel_size, min_screen_surfel_size, max_screen_surfel_size);

	 //    VertexOut.pass_point_color = DataPointToColor(screen_surfel_size, min_screen_surfel_size, max_screen_surfel_size);
  //       break;
  //       }
  //    case 2:
  //       {
  //        vec3 provenance_normal = VertexOut.pass_normal;
  //        if( provenance_normal.z < 0 )
  //           provenance_normal = provenance_normal * -1; 

  //        //VertexOut.pass_point_color = vec3(in_normal * 0.5 + 0.5);
  //        VertexOut.pass_point_color = vec3(provenance_normal * 0.5 + 0.5);
  //       }
	 // break;
  //    case 3:
  //       {
  //        VertexOut.pass_point_color = VertexOut.pass_point_color + vec3(accuracy, 0.0, 0.0);
  //       }
	 // break;
  //    default:
  //       break;
  // }

}
