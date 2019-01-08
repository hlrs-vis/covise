// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_MEASUREMENT_H_
#define REN_MEASUREMENT_H_

#include <scm/core.h>
#include <scm/log.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>
#include <scm/core/pointer_types.h>
#include <scm/core/io/tools.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/gl_util/data/imaging/texture_loader.h>
#include <scm/gl_util/viewer/camera.h>
#include <scm/gl_util/primitives/quad.h>
#include <scm/gl_util/primitives/box.h>

#include <scm/core/math.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/primitives/geometry.h>

#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/font/text.h>
#include <scm/gl_util/font/text_renderer.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>
#include <scm/gl_util/primitives/geometry.h>


#include <vector>


class Measurement{

 public:
  Measurement();
  ~Measurement();
  
  void drawInfo(scm::shared_ptr<scm::gl::render_device> device,
		scm::shared_ptr<scm::gl::render_context> context,
		scm::gl::text_renderer_ptr text_renderer,
		scm::gl::text_ptr renderable_text,
		int screen_width, int screen_height,
		scm::math::mat4f projection_matrix,
		scm::math::mat4f view_matrix, bool do_measurement, bool display_info);

  void mouse(scm::shared_ptr<scm::gl::render_device> device,
	     int button, int state, int mouse_h, int mouse_v,
	     int screen_width, int screen_height,
	     scm::math::mat4f projection_matrix,
	     scm::math::mat4f view_matrix);

 private:

  std::vector<scm::math::vec4f> pick_positions_;



};


#endif
