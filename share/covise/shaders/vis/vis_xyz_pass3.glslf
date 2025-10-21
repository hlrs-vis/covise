// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core
 
layout(binding  = 0) uniform sampler2D in_color_texture;

OPTIONAL_BEGIN
  layout(binding  = 1) uniform sampler2D in_normal_texture;
  layout(binding  = 2) uniform sampler2D in_vs_position_texture;
OPTIONAL_END

layout(location = 0) out vec4 out_color;

uniform vec3 background_color;
   

in vec2 pos;


OPTIONAL_BEGIN
  // shading
  INCLUDE ../common/shading/blinn_phong.glsl
OPTIONAL_END

void main()	{

    out_color = vec4(background_color.r,background_color.g,background_color.b, 1.0f);

    vec4 texColor = texture2D(in_color_texture, (pos.xy + 1) / 2.0f);
	
    // w contains the accumulated weights that can be shared over several
    // attributes at the same surfel position
    if(texColor.w != 0.0f) {
      texColor.xyz = (texColor.xyz/texColor.w);
      
      vec4 color_to_write = vec4(texColor.xyz, 1.0);

      //optional code for looking up shading attributes and performing shading
    OPTIONAL_BEGIN
      //optional
      vec3 texNormal     = texture2D(in_normal_texture, (pos.xy + 1) / 2.0f).xyz;
      vec3 texVSPosition = texture2D(in_vs_position_texture, (pos.xy + 1) / 2.0f).xyz;
      texNormal = texNormal/texColor.w;
      texVSPosition.xyz = texVSPosition.xyz/texColor.w;

      vec3 shaded_color = shade_blinn_phong(texVSPosition, texNormal, vec3(0.0, 0.0, 0.0), texColor.rgb);

      color_to_write = vec4(shaded_color,  1.0);
    OPTIONAL_END

      out_color = color_to_write;
    }


 }
