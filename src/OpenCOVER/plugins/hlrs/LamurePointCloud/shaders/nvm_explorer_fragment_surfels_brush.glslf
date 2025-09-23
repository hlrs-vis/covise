// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

uniform mat4 model_view_matrix;
uniform vec3 color_brush_surfels;

in VertexData
{
    vec3 pass_position;
    vec3 pass_normal;
    vec2 pass_uv_coords;
}
VertexIn;

layout(location = 0) out vec4 out_color;

void main()
{
    vec2 uv_coords = VertexIn.pass_uv_coords;

    if(dot(uv_coords, uv_coords) > 1)
    {
        discard;
    }

    vec3 position_space_view = (model_view_matrix * vec4(VertexIn.pass_position, 1.0)).xyz;
    vec3 position_space_view_normalized = normalize(position_space_view);

    // vec4 normal = transpose(inverse(model_view_matrix)) * vec4(VertexIn.pass_normal, 1.0);

    vec4 normal_space_view = model_view_matrix * vec4(VertexIn.pass_normal, 0.0);
    // vec4 normal_space_view = model_view_matrix * vec4(normalize(normal.xyz), 1.0);
    vec3 normal_space_view_normalized = normalize(normal_space_view.xyz);

    float dot_product = dot(-position_space_view_normalized, normal_space_view_normalized);

    // if(dot_product < 0.0f)
    // {
    //     discard;
    // }

    dot_product = (dot_product + 1.0) * 0.5;

    out_color = vec4(color_brush_surfels * dot_product, 1.0);
}
