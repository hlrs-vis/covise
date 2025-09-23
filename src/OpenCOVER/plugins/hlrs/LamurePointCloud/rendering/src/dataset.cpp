// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/dataset.h>

#include <scm/core/math.h>

namespace lamure
{

namespace ren
{

dataset::
dataset(const std::string& filename)
: model_id_(invalid_model_t),
  is_loaded_(false),
  bvh_(nullptr),
  transform_(scm::math::mat4f::identity()) {
    load(filename);
}

dataset::
~dataset() {
    if (bvh_ != nullptr) {
        delete bvh_;
        bvh_ = nullptr;
    }
}

void dataset::
load(const std::string& filename) {

    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    
    if (extension.compare("bvhqz") == 0 || extension.compare("bvh") == 0) {
        bvh_ = new bvh(filename);
        is_loaded_ = true;
    }
    else {
        throw std::runtime_error(
            "lamure: dataset::Incompatible input file: " + filename);
    }
    
}


} // namespace ren

} // namespace lamure


