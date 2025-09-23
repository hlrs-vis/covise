// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/ren/policy.h>

#include <lamure/ren/controller.h>

namespace lamure
{

namespace ren
{

std::mutex policy::mutex_;
bool policy::is_instanced_ = false;
policy* policy::single_ = nullptr;

policy::
policy()
: reset_system_(true),
  max_upload_budget_in_mb_(LAMURE_DEFAULT_UPLOAD_BUDGET),
  render_budget_in_mb_(LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET),
  out_of_core_budget_in_mb_(LAMURE_DEFAULT_MAIN_MEMORY_BUDGET),
  size_of_provenance_(LAMURE_DEFAULT_SIZE_OF_PROVENANCE),
  window_width_(800),
  window_height_(600) {

}

policy::
~policy() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_instanced_ = false;
}

policy* policy::
get_instance() {
    if (!is_instanced_) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!is_instanced_) {
            single_ = new policy();
            is_instanced_ = true;
        }

        return single_;
    }
    else {
        return single_;
    }
}

} // namespace ren

} // namespace lamure


