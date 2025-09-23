// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_MANAGEMENT_ID_HISTOGRAM_RENDERER_CORNERS_H_
#define LAMURE_PVS_MANAGEMENT_ID_HISTOGRAM_RENDERER_CORNERS_H_

#include <lamure/pvs/pvs_preprocessing.h>
#include "lamure/pvs/management_base.h"

namespace lamure
{
namespace pvs
{

class management_id_histogram_renderer_corners : public management_base
{
public:
                        management_id_histogram_renderer_corners(std::vector<std::string> const& model_filenames,
                                                        		std::vector<scm::math::mat4f> const& model_transformations,
                                                        		const std::set<lamure::model_t>& visible_set,
                                                        		const std::set<lamure::model_t>& invisible_set);
                        
    virtual             ~management_id_histogram_renderer_corners();

                        management_id_histogram_renderer_corners(const management_id_histogram_renderer_corners&) = delete;
                        management_id_histogram_renderer_corners& operator=(const management_id_histogram_renderer_corners&) = delete;

    virtual bool        MainLoop();

protected:

	bool				first_phase_;
	scm::math::vec3d	smallest_cell_size_;
	size_t				current_corner_index_x_, current_corner_index_y_, current_corner_index_z_;
};

}
}

#endif
