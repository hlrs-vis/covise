// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#version 420 core

layout(early_fragment_tests) in;

const float gaussian[32] = float[](
  1.000000, 1.000000, 0.988235, 0.968627, 0.956862, 0.917647, 0.894117, 0.870588, 0.915686, 0.788235,
  0.749020, 0.690196, 0.654902, 0.619608, 0.552941, 0.513725, 0.490196, 0.458824, 0.392157, 0.356863,
  0.341176, 0.278431, 0.254902, 0.227451, 0.188235, 0.164706, 0.152941, 0.125490, 0.109804, 0.098039,
  0.074510, 0.062745
);

in VertexData {
  //output to fragment shader
  vec3 pass_point_color;
  vec3 pass_normal;
  vec2 pass_uv_coords;
  OPTIONAL_BEGIN
    vec3 mv_vertex_position;
  OPTIONAL_END
} VertexIn;


layout(location = 0) out vec4 accumulated_colors;

OPTIONAL_BEGIN
  layout(location = 1) out vec3 accumulated_normals;
  layout(location = 2) out vec3 accumulated_vs_positions;
OPTIONAL_END

uniform vec2 win_size;

void main() {
  vec2 uv_coords = VertexIn.pass_uv_coords;

  if ( dot(uv_coords, uv_coords) > 1 )
    discard;

  vec3 normal = VertexIn.pass_normal;

  if( normal.z < 0 )
    normal = normal * -1; 

  normal = (normal + vec3(1.0, 1.0, 1.0)) / 2.0;
  float weight = gaussian[int(round(length(uv_coords) * 31.0 ))];

  accumulated_colors = vec4(VertexIn.pass_point_color * weight, weight);


  OPTIONAL_BEGIN
    vec3 adjustedNormal = vec3(0.0,0.0,0.0);
    if (VertexIn.pass_normal.z < 0) {
      adjustedNormal = VertexIn.pass_normal.xyz * -1;
    }
    else {
      adjustedNormal = VertexIn.pass_normal.xyz;
    }
    accumulated_normals = vec3(adjustedNormal.xyz * weight);
    accumulated_vs_positions = vec3(VertexIn.mv_vertex_position.xyz * weight);
  OPTIONAL_END



}

