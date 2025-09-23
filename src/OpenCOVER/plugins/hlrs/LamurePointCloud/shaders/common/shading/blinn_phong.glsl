// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

uniform int use_material_color;
uniform vec3 material_diffuse;
uniform vec4 material_specular;
uniform vec3 ambient_light_color;
uniform vec4 point_light_color;

// blinn-phong-shading in view space, => camera in 0,0,0
vec3 shade_blinn_phong(in vec3 vs_pos, in vec3 vs_normal, 
                       in vec3 vs_light_pos, in vec3 in_col) {

  vec3 light_dir = (vs_light_pos-vs_pos);
  float light_distance = length(light_dir);
  light_dir /= light_distance;

  light_distance = light_distance * light_distance;

  float NdotL = dot(vs_normal, light_dir);
  float diffuse_intensity =  max(0.0, NdotL);

  vec3 view_dir = (-vs_pos);

  // due to the normalize function's reciprocal square root
  vec3 H = normalize( light_dir + view_dir );

    //Intensity of the specular light
  float NdotH = dot( vs_normal, H );

  float m = material_specular.a;

  float specular_intensity = pow( max(NdotH, 0.0), m );

  
  vec3 albedo = in_col;

  if(1 == use_material_color) {
    albedo = material_diffuse;
  }

  return 
    ambient_light_color.rgb + 
    diffuse_intensity * point_light_color.rgb * albedo * point_light_color.a +
    specular_intensity * point_light_color.rgb * material_specular.rgb * point_light_color.a;
}