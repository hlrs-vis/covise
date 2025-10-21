// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr


uniform int render_provenance;
uniform float average_radius;
uniform float accuracy;

//exposes "data_value_to_rainbow"
INCLUDE heatmapping/wavelength_to_rainbow.glsl

// "public" interface to be called in the vertex shader
void resolve_provenance_coloring(in vec3 position, in vec3 normal, in vec3 tangent, in vec3 bitangent, 
                                 in float radius, in out vec3 color ) {

  switch(render_provenance) {
     case 1: { 
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

      color = data_value_to_rainbow(screen_surfel_size, min_screen_surfel_size, max_screen_surfel_size);
      break;
    }
     
    case 2: {
      vec3 provenance_normal = normal;
      if( provenance_normal.z < 0 ) {
        provenance_normal = provenance_normal * -1;
      }

        color = vec3(normal * 0.5 + 0.5);
        color = vec3(provenance_normal * 0.5 + 0.5);

      break;
    }

    
    case 3: {
      color = color + vec3(accuracy, 0.0, 0.0);

      break;
    }

     default: {
        break;
    }
     
  }

}