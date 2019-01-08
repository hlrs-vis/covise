// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_UTILS_H_
#define REN_UTILS_H_

#include <scm/core/math.h>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>
#include <lamure/types.h>


std::pair< std::vector<std::string>, std::vector<scm::math::mat4f> > read_model_string(std::string const& path_to_resource_file, std::set<lamure::model_t>* visible_set, std::set<lamure::model_t>* invisible_set);

void create_scene_name_from_vector(std::vector<std::string> const&, std::string&);

void create_scene_name_from_camera_session_file(std::string const& session_file, std::string& name);

#endif
