// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core
 
layout(binding = 0) uniform usampler2D in_color_texture;

layout(location = 0) out vec4 out_color;

in vec2 pos;


void main()	
{
	uvec4 texColor = texture(in_color_texture, (pos.xy + 1) / 2.0f);	
	//uvec4 texColor = texelFetch(in_color_texture, ivec2(gl_FragCoord.xy), 0);
	texColor.w = 255;
	
    out_color = vec4(texColor) / 255.0;
}
