// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "measurement.h"

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/math.h>
#include <scm/core/math.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <iostream>
#include <sstream>
#include <iomanip>

#define  GLUT_LEFT_BUTTON                   0x0000
#define  GLUT_MIDDLE_BUTTON                 0x0001
#define  GLUT_RIGHT_BUTTON                  0x0002
#define  GLUT_DOWN                          0x0000
#define  GLUT_UP                            0x0001

Measurement::Measurement()
  : pick_positions_()
{}


Measurement::~Measurement()
{}




void
Measurement::drawInfo(scm::shared_ptr<scm::gl::render_device> device,
		      scm::shared_ptr<scm::gl::render_context> context,
		      scm::gl::text_renderer_ptr text_renderer,
		      scm::gl::text_ptr renderable_text,
		      int screen_width, int screen_height,
		      scm::math::mat4f projection_matrix,
		      scm::math::mat4f view_matrix, bool do_measurement, bool display_info){




  const scm::math::mat4f viewport_scale = scm::math::make_scale(screen_width * 0.5f, screen_height * 0.5f, 0.5f);;
  const scm::math::mat4f viewport_translate = scm::math::make_translation(1.0f,1.0f,1.0f);
  const scm::math::mat4f world_to_image =  viewport_scale * viewport_translate * projection_matrix * view_matrix;
  for(unsigned i = 0; i != pick_positions_.size(); ++i){
    const scm::math::vec4f pos_world = pick_positions_[i];
    scm::math::vec4f pos_image = world_to_image * pos_world;
    pos_image[0] /= pos_image[3];
    pos_image[1] /= pos_image[3];
    pos_image[2] /= pos_image[3];
    pos_image[3] /= pos_image[3];
    //std::cout << "pos_image: " << pos_image << std::endl;

    std::stringstream os;
    os << "+" << (i + 1) << "(" << pos_world[0] << ", " << pos_world[1] << ", " << pos_world[2] << ")";
    renderable_text->text_string(os.str());
    text_renderer->draw_shadowed(context, scm::math::vec2i( (unsigned int) pos_image[0] - 4, (unsigned int) pos_image[1]) - 7, renderable_text);

    if(i > 0){
      const scm::math::vec4f pos_world_last = pick_positions_[i - 1];
      const float dist = scm::math::length(scm::math::vec3f(pos_world[0], pos_world[1], pos_world[2]) - 
					   scm::math::vec3f(pos_world_last[0], pos_world_last[1], pos_world_last[2]));
      
      std::stringstream os_m;
      if(dist < 1.0f){
	os_m << "[" << (i) << ", " << i + 1 << "]: " << int(dist * 1000.0f) << " mm";
      }
      else{
	os_m << "[" << (i) << ", " << i + 1 << "]: " << std::setprecision(4) << dist << " m";
      }
      renderable_text->text_string(os_m.str());
      scm::math::vec4f pos_image = world_to_image * ((pos_world_last + pos_world) * 0.5);
      pos_image[0] /= pos_image[3];
      pos_image[1] /= pos_image[3];
      pos_image[2] /= pos_image[3];
      pos_image[3] /= pos_image[3];
      text_renderer->draw_shadowed(context, scm::math::vec2i( (unsigned int) pos_image[0] - 4, (unsigned int) pos_image[1]) - 7, renderable_text);
    }

  }

  if(display_info){
    std::stringstream os;
    if(do_measurement){
      os << "3D measurement enabled: Right click to add a new measurement point, middle click to delete last measurement point\n";
    }
    else{
      os << "Press m for 3D measurement\n";
    }
    renderable_text->text_string(os.str());
    text_renderer->draw_shadowed(context, scm::math::vec2i( 20, 40), renderable_text);
  }
}


void
Measurement::mouse(scm::shared_ptr<scm::gl::render_device> device,
		   int button, int state, int mouse_h, int mouse_v,
		   int screen_width, int screen_height,
		   scm::math::mat4f projection_matrix,
		   scm::math::mat4f view_matrix){


  const unsigned screen_x = mouse_h;
  const unsigned screen_y = screen_height - mouse_v;




  if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {


    //std::cout << "button: " << button << " state: " << state << " x: " << screen_x << " y: " << screen_y << std::endl;
    //std::cout << "projection_matrix: " << projection_matrix << std::endl;
    //std::cout << "view_matrix: " << view_matrix << std::endl;

    float screen_depth;
    device->opengl_api().glReadPixels( screen_x, screen_y, 1, 1,
				       GL_DEPTH_COMPONENT, GL_FLOAT, &screen_depth);
    
    //std::cout << "screen_depth: " << screen_depth << std::endl;

    if(screen_depth < 1.0f){
      const scm::math::mat4f viewport_scale = scm::math::make_scale(screen_width * 0.5f, screen_height * 0.5f, 0.5f);;
      const scm::math::mat4f viewport_translate = scm::math::make_translation(1.0f,1.0f,1.0f);
      const scm::math::mat4f eye_to_image =  viewport_scale * viewport_translate * projection_matrix;
      const scm::math::mat4f image_to_eye = scm::math::inverse(eye_to_image);
      scm::math::vec4f pos_eye = image_to_eye * scm::math::vec4f(screen_x + 0.5f, screen_y + 0.5f, screen_depth, 1.0);
      //std::cout << "pos_eye: " << pos_eye << std::endl;
      pos_eye[0] /= pos_eye[3];
      pos_eye[1] /= pos_eye[3];
      pos_eye[2] /= pos_eye[3];
      pos_eye[3] /= pos_eye[3];
      //std::cout << "pos_eye: " << pos_eye << std::endl;
      const scm::math::mat4f eye_to_world =  scm::math::inverse(view_matrix);
      pick_positions_.push_back(eye_to_world * pos_eye);
      //scm::math::mat4f blub_matrix = scm::math::inverse(blub_matrix);
    }
  }

  if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN) {
    pick_positions_.pop_back();
  }
}

