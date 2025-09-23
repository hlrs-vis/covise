// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_VISIBILITY_TEST_ID_HISTOGRAM_RENDERER_H
#define LAMURE_PVS_VISIBILITY_TEST_ID_HISTOGRAM_RENDERER_H

#include <lamure/pvs/pvs_preprocessing.h>
#include "lamure/pvs/visibility_test.h"
#include "lamure/pvs/management_id_histogram_renderer.h"
#include "lamure/pvs/grid.h"

namespace lamure
{
namespace pvs
{

class visibility_test_id_histogram_renderer : public visibility_test
{
public:
	visibility_test_id_histogram_renderer();
	virtual ~visibility_test_id_histogram_renderer();

	virtual int initialize(int& argc, char** argv);
	virtual void test_visibility(grid* visibility_grid);
	virtual void shutdown();

	virtual bounding_box get_scene_bounds() const;

private:
	int resolution_x_;
	int resolution_y_;
	unsigned int main_memory_budget_;
    unsigned int video_memory_budget_;
    unsigned int max_upload_budget_;

    management_id_histogram_renderer* management_;
    bounding_box scene_bounds_;
};

}
}

#endif
