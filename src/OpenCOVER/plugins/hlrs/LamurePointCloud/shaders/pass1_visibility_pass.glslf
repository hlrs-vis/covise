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

void main()
{
  vec2 uv_coords = VertexIn.pass_uv_coords;

  if( dot(uv_coords, uv_coords) >  1 )
    discard;

  vec2 mappedPointCoord = gl_PointCoord*2 + vec2(-1.0, -1.0);

  gl_FragDepth = gl_FragCoord.z + (VertexIn.pass_es_shift / far_plane);
}

