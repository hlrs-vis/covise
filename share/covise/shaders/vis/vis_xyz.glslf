// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

in VertexData {
    vec3 pass_point_color;
    vec3 pass_normal;
    vec2 pass_uv_coords;
    OPTIONAL_BEGIN
      vec3 mv_vertex_position;
    OPTIONAL_END
} VertexIn;

layout(location = 0) out vec4 out_color;

OPTIONAL_BEGIN
  INCLUDE ../common/shading/blinn_phong.glsl
OPTIONAL_END

void main()
{

  vec2 uv_coords = VertexIn.pass_uv_coords;

  if ( dot(uv_coords, uv_coords) > 1.0 ) {
    discard;
    //out_color = vec4(1.0, 0.0, 0.0, 1.0);
  }
  else {

    vec4 color_to_write = vec4(VertexIn.pass_point_color.xyz, 1.0);

    //optional code for looking up shading attributes and performing shading
    OPTIONAL_BEGIN
      vec3 adjustedNormal = vec3(0.0,0.0,0.0);
      if (VertexIn.pass_normal.z < 0) {
	adjustedNormal = VertexIn.pass_normal.xyz * -1;
      }
      else {
        adjustedNormal = VertexIn.pass_normal.xyz;
      }
      vec3 shaded_color = shade_blinn_phong(VertexIn.mv_vertex_position, adjustedNormal, 
                                          vec3(0.0, 0.0, 0.0), color_to_write.rgb);
      color_to_write = vec4(shaded_color,  1.0);
    OPTIONAL_END

    out_color = color_to_write;
  }

  //out_color = vec4(1.0, 0.0, 0.0, 1.0);
}

