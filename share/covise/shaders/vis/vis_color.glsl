// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

INCLUDE ../common/heatmapping/wavelength_to_rainbow.glsl
INCLUDE ../common/heatmapping/colormap.glsl

uniform bool show_normals;
uniform bool show_accuracy;
uniform bool show_radius_deviation;
uniform bool show_output_sensitivity;
uniform float accuracy;

uniform float average_radius;

uniform int channel;
uniform bool heatmap;
uniform float heatmap_min;
uniform float heatmap_max;
uniform vec3 heatmap_min_color;
uniform vec3 heatmap_max_color;

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
  ms_u = normalize(tmp_ms_u) * point_size_factor  * radius;
  ms_v = normalize(cross(ms_n, tmp_ms_u)) * point_size_factor * radius;
}


vec3 quick_interp(vec3 color1, vec3 color2, float value) {
  return color1 + (color2 - color1) * clamp(value, 0, 1);
}

vec3 get_output_sensitivity_color(in vec3 position, in vec3 normal, in float radius) {
  vec3 tangent   = vec3(0.0);
  vec3 bitangent = vec3(0.0);
  compute_tangent_vectors(normal, radius, tangent, bitangent);

  // finalize normal and tangents
  //vec3 view_normal = normalize((inv_mv_matrix * vec4(normal, 0.0)).xyz );

  // finalize color with provenance overlay

  float ideal_screen_surfel_size = 2.0; // error threshold
  float min_screen_surfel_size = 0.0; // error threshold
  float max_screen_surfel_size = 10.0; // error threshold
  
  vec4 surfel_pos_screen = model_to_screen_matrix * vec4(position ,1.0);
       surfel_pos_screen /= surfel_pos_screen.w;
  vec4 border_pos_screen_u = model_to_screen_matrix * vec4(position + tangent, 1.0);
       border_pos_screen_u /= border_pos_screen_u.w;
  vec4 border_pos_screen_v = model_to_screen_matrix * vec4(position + bitangent, 1.0);
       border_pos_screen_v /= border_pos_screen_v.w;
  float screen_surfel_size = max(length(surfel_pos_screen.xy - border_pos_screen_u.xy), length(surfel_pos_screen.xy - border_pos_screen_v.xy));
        screen_surfel_size = clamp(screen_surfel_size, min_screen_surfel_size, max_screen_surfel_size);

  return data_value_to_rainbow(screen_surfel_size, min_screen_surfel_size, max_screen_surfel_size);

}

vec3 get_color(in vec3 position, in vec3 normal, in vec3 color, in float radius) {

  float prov_value = 0.0;
  vec3 view_color = vec3(0.0);

  if (show_normals) {
    vec4 vis_normal = model_radius_scale * vec4(normal, 0.0);
    if( vis_normal.z < 0 ) {
      vis_normal = vis_normal * -1;
    }
    view_color = vec3(vis_normal.xyz * 0.5 + 0.5);
  }
  else if (show_output_sensitivity) {
    view_color = get_output_sensitivity_color(position, normal, radius);
  }
  else if (show_radius_deviation) {
    float max_fac = 2.0;
    view_color = vec3(min(max_fac, radius/average_radius) / max_fac);
  }
  else if (channel == 0) {
    view_color = color;
  }
  else {
    if (channel == 1) {
      prov_value = prov1;
    }
    else if (channel == 2) {
      prov_value = prov2;
    }
    else if (channel == 3) {
      prov_value = prov3;
    }
    else if (channel == 4) {
      prov_value = prov4;
    }
    else if (channel == 5) {
      prov_value = prov5;
    }
    else if (channel == 6) {
      prov_value = prov6;
    }
    if (heatmap) {
      float value = (prov_value - heatmap_min) / (heatmap_max - heatmap_min);
      view_color = quick_interp(heatmap_min_color, heatmap_max_color, value);
    }
    else {
      init_colormap();
      float value = (prov_value - heatmap_min) / (heatmap_max - heatmap_min);
      view_color = get_colormap_value(value);
    }
  }

  if (show_accuracy) {
    view_color = view_color + vec3(accuracy, 0.0, 0.0);
  }

  return view_color;

}

