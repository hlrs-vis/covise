// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

void compute_tangent_vectors(in vec3 normal, in float radius, out vec3 ms_u, out vec3 ms_v) {

  vec3 ms_n = normalize(normal.xyz);
  vec3 tmp_ms_u = vec3(0.0);

  // compute arbitrary tangent vectors
  if(ms_n.z != 0.0) {
    tmp_ms_u = vec3( 1, 1, (-ms_n.x -ms_n.y)/ms_n.z);
  } else if (ms_n.y != 0.0) {
    tmp_ms_u = vec3( 1, (-ms_n.x -ms_n.z)/ms_n.y, 1);
  } else {
    tmp_ms_u = vec3( (-ms_n.y -ms_n.z)/ms_n.x, 1, 1);
  }

  // assign tangent vectors
  ms_u = normalize(tmp_ms_u) * point_size_factor * model_radius_scale * radius;
  ms_v = normalize(cross(ms_n, tmp_ms_u)) * point_size_factor * model_radius_scale * radius;
}

