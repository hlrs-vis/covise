// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core
 
layout(binding  = 0) uniform sampler2D in_color_texture;
layout(location = 0) out vec4 out_color;
        
//for texture access
in vec2 pos;


void main()	
{
  //vec4 texColor = texture2D(in_color_texture, (pos.xy + 1) / 2.0f);
  
  vec4 texColor = texture2D(in_color_texture, (pos.xy + 1) / 2.0f);

  out_color = vec4(texColor.xyz, 1.0f);  

 }
