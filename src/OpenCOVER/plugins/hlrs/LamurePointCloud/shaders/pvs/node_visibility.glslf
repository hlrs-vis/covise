// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

in VertexData {
  vec2 pass_uv_coords;
  float pass_es_shift;
} VertexIn;

uniform float far_plane;
uniform int model_id = 0;
uniform int node_id = 0;

// Color rendered as output containing node and model ID.
layout(location = 0) out uvec4 out_color_int;

// Remodel IDs to fit output color.
uvec4 color_from_id(int node_id, int model_id)
{
    return uvec4(((node_id) & 0xFF), ((node_id >> 8) & 0xFF), ((node_id >> 16) & 0xFF), (model_id & 0xFF));

    /*if(node_id % 3 == 0)
        return uvec4(((node_id >> 16) & 0xFF), ((node_id >> 8) & 0xFF), (node_id & 0xFF), (model_id & 0xFF));
    else if (node_id % 3 == 1)
        return uvec4((node_id & 0xFF), ((node_id >> 8) & 0xFF), ((node_id >> 16) & 0xFF), (model_id & 0xFF));
    else
        return uvec4(((node_id >> 8) & 0xFF), (node_id & 0xFF), ((node_id >> 16) & 0xFF), (model_id & 0xFF));*/
}

void main()
{
  vec2 uv_coords = VertexIn.pass_uv_coords;

  if( dot(uv_coords, uv_coords) > 1 )
    discard;

  vec2 mappedPointCoord = gl_PointCoord*2 + vec2(-1.0f, -1.0f);

  gl_FragDepth = gl_FragCoord.z + (VertexIn.pass_es_shift / far_plane);
  
  out_color_int = color_from_id(node_id, 255 - model_id);
}

