// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_MANAGEMENT_SIMPLE_RANDOMIZED_H_
#define LAMURE_PVS_MANAGEMENT_SIMPLE_RANDOMIZED_H_

#include <chrono>
#include <random>

#include <lamure/pvs/pvs_preprocessing.h>
#include "lamure/pvs/management_base.h"

namespace lamure
{
namespace pvs
{

class management_simple_randomized : public management_base
{
public:
                        management_simple_randomized(std::vector<std::string> const& model_filenames,
                            std::vector<scm::math::mat4f> const& model_transformations,
                            const std::set<lamure::model_t>& visible_set,
                            const std::set<lamure::model_t>& invisible_set);
                        
    virtual             ~management_simple_randomized();

                        management_simple_randomized(const management_simple_randomized&) = delete;
                        management_simple_randomized& operator=(const management_simple_randomized&) = delete;

    virtual bool        MainLoop();

    void                set_duration_visibility_test(const double& duration);
    void                set_samples_visibility_test(const size_t& num_samples);

protected:
    
    double duration_visibility_test_in_seconds_;
    double remaining_duration_visibility_test_in_seconds_;
    double skipped_duration_visibility_test_in_seconds_;
    std::chrono::time_point<std::chrono::system_clock> current_test_start_time_;

    size_t num_samples_to_finish_;
    size_t num_samples_completed_;

    // Used to manage the random components of the visibility test.
    std::mt19937 view_cell_rng_;
    std::uniform_int_distribution<size_t> view_cell_distribution_;

    scm::math::vec3d current_position_in_current_view_cell_;

    std::vector<size_t> num_samples_per_view_cell_;
};

}
}

#endif
