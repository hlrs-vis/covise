// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_MANAGEMENT_ID_HISTOGRAM_RENDERER_H_
#define LAMURE_PVS_MANAGEMENT_ID_HISTOGRAM_RENDERER_H_

#include <lamure/pvs/pvs_preprocessing.h>
#include "lamure/pvs/management_base.h"

namespace lamure
{
namespace pvs
{

class management_id_histogram_renderer : public management_base
{
public:
                        management_id_histogram_renderer(std::vector<std::string> const& model_filenames,
                                                        std::vector<scm::math::mat4f> const& model_transformations,
                                                        const std::set<lamure::model_t>& visible_set,
                                                        const std::set<lamure::model_t>& invisible_set);
                        
    virtual             ~management_id_histogram_renderer();

                        management_id_histogram_renderer(const management_id_histogram_renderer&) = delete;
                        management_id_histogram_renderer& operator=(const management_id_histogram_renderer&) = delete;

    virtual bool        MainLoop();
};

}
}

#endif
