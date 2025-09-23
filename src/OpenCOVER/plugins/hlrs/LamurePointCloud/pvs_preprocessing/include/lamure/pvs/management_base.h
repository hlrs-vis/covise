// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_PVS_MANAGEMENT_BASE_H_
#define LAMURE_PVS_MANAGEMENT_BASE_H_

#include <lamure/pvs/pvs_preprocessing.h>
#include <lamure/types.h>

#include "lamure/pvs/glut_management.h"
#include "lamure/pvs/renderer.h"
#include "lamure/pvs/grid.h"

#include <lamure/ren/config.h>
#include <lamure/ren/model_database.h>
#include <lamure/ren/controller.h>
#include <lamure/ren/cut.h>
#include <lamure/ren/cut_update_pool.h>

#include <GL/freeglut.h>
#include <FreeImagePlus.h>
#include <lamure/ren/ray.h>

//#define ALLOW_INPUT
//#define LAMURE_PVS_USE_AS_RENDERER
#define LAMURE_PVS_MEASURE_PERFORMANCE          // Will output some info on runtime in the terminal.

namespace lamure
{
namespace pvs
{

class management_base : public glut_management
{
public:
                        management_base(std::vector<std::string> const& model_filenames,
                            			std::vector<scm::math::mat4f> const& model_transformations,
                            			const std::set<lamure::model_t>& visible_set,
                            			const std::set<lamure::model_t>& invisible_set);
                        
	virtual             ~management_base();

                        management_base(const management_base&) = delete;
                        management_base& operator=(const management_base&) = delete;

    virtual bool        MainLoop() = 0;
    void                update_trackball(int x, int y);
    void                RegisterMousePresses(int button, int state, int x, int y);
    void                dispatchKeyboardInput(unsigned char key);
    void                dispatchResize(int w, int h);

    void                SetSceneName();

    float               error_threshold_;
    void                IncreaseErrorThreshold();
    void                DecreaseErrorThreshold();

    void                set_grid(grid* visibility_grid);
    void                set_pvs_file_path(const std::string& file_path);

protected:

    void                Toggledispatching();

    void                check_for_nodes_within_cells(const std::vector<std::vector<size_t>>& total_depths, const std::vector<std::vector<size_t>>& total_nums);

    void                emit_node_visibility(grid* visibility_grid);
    void                set_node_parents_visible(const size_t& cell_id, const view_cell* cell, const model_t& model_id, const node_t& node_id);
    void                set_node_children_visible(const size_t& cell_id, const view_cell* cell, const model_t& model_id, const node_t& node_id);

    void                apply_temporal_pvs(const id_histogram& hist);

    lamure::ren::camera::mouse_state mouse_state_;

    Renderer* renderer_;

    lamure::ren::camera*   active_camera_;

    int32_t             width_;
    int32_t             height_;

    float importance_;
    float visibility_threshold_;

    bool test_send_rendered_;

    bool                dispatch_;

    lamure::model_t     num_models_;

    float               near_plane_;
    float               far_plane_;

    std::vector<scm::math::mat4f> model_transformations_;
    std::vector<std::string> model_filenames_;

    bool                first_frame_;

    grid*               visibility_grid_;
    size_t              current_grid_index_;
    unsigned short      direction_counter_;

    bool                update_position_for_pvs_;

    std::string         pvs_file_path_;

#ifdef LAMURE_PVS_MEASURE_PERFORMANCE
    // Used to measure performance of preprocessing substeps.
    double              total_cut_update_time_;
    double              total_render_time_;
    double              total_histogram_evaluation_time_;
#endif

    // Used to identify the depth of nodes for the check which nodes are inside the grid cells. (model id<grid cell id<data>>)
    std::vector<std::vector<size_t>> total_depth_rendered_nodes_;
    std::vector<std::vector<size_t>> total_num_rendered_nodes_;
};

}
}

#endif
