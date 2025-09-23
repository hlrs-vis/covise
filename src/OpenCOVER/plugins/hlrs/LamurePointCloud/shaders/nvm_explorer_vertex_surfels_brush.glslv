// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

uniform float;
uniform float point_size_factor;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

out VertexData
{
    vec3 position;
    vec3 normal;
    vec3 pass_ms_u;
    vec3 pass_ms_v;
}
VertexOut;

void main()
{
    VertexOut.position = in_position;
    VertexOut.normal = in_normal;

    float in_radius = 1.0;

    vec3 ms_n = normalize(in_normal.xyz);
    vec3 ms_u;

    //**compute tangent vectors**//
    if(ms_n.z != 0.0)
    {
        ms_u = vec3(1, 1, (-ms_n.x - ms_n.y) / ms_n.z);
    }
    else if(ms_n.y != 0.0)
    {
        ms_u = vec3(1, (-ms_n.x - ms_n.z) / ms_n.y, 1);
    }
    else
    {
        ms_u = vec3((-ms_n.y - ms_n.z) / ms_n.x, 1, 1);
    }

    //**assign tangent vectors**//
    VertexOut.pass_ms_u = normalize(ms_u) * point_size_factor * in_radius;
    VertexOut.pass_ms_v = normalize(cross(ms_n, ms_u)) * point_size_factor * in_radius;

    gl_Position = vec4(in_position, 1.0);
}
