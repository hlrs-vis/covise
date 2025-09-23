// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core
 
layout(location = 0) out vec4 out_color;
       
uniform int culling_status;

void main()
{
	if(culling_status == 0)  //inside
	{
 	   out_color = vec4(0.0f, 1.0f, 0.0f, 1.0f);
	}
	else if(culling_status == 2) 
	{
	   out_color = vec4(1.0f, 1.0f, 0.0f, 1.0f);
	}
	else if(culling_status == 1)
	{
	   out_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	}
	else
	{
	   out_color = vec4(1.0f, 0.0f, 1.0f, 1.0f);
	}
}
