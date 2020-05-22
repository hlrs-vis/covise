/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "LamureDrawable.h"
#include "Lamure.h"
#include "management.h"
#include <cover/coVRConfig.h>

LamureDrawable::ContextState::ContextState()
{
}

LamureDrawable::ContextState::~ContextState()
{
}
LamureDrawable::LamureDrawable()
{
#ifdef VERBOSE
    cerr << "VolumeDrawable::<init> warn: empty constructor called" << endl;
#endif
    init();
}

LamureDrawable::LamureDrawable(const LamureDrawable &drawable,
    const osg::CopyOp &copyop)
    : Drawable(drawable, copyop)
{
#ifdef VERBOSE
    cerr << "VolumeDrawable::<init> copying" << endl;
#endif
    init();
}

void LamureDrawable::init()
{
    setSupportsDisplayList(false);
}


LamureDrawable::~LamureDrawable()
{
#ifdef VERBOSE
    cerr << "VolumeDrawable::<dtor>: this=" << this << endl;
#endif
    contextState.clear();
}

int LamureDrawable::loadBVH(const char *filename)
{

    model_transformations.push_back(scm::math::mat4f::identity());
    if (management_ == nullptr)
    {
        std::string name(filename);
        model_filenames.push_back(name);
    }
    else
    {
        lamure::ren::model_database* database = lamure::ren::model_database::get_instance();

        lamure::model_t model_id = database->add_model(filename, std::to_string(numModels));
        ++numModels;

        float scene_diameter = coVRConfig::instance()->farClip();
        osg::BoundingBox bbox;
        //scm::gl::box_impl<float>::min_vertex
        for (lamure::model_t model_id = 0; model_id < database->num_models(); ++model_id)
        {
            const auto &bb = database->get_model(model_id)->get_bvh()->get_bounding_boxes()[0];
            bbox.expandBy(bb.min_vertex().x, bb.min_vertex().y, bb.min_vertex().z);
            bbox.expandBy(bb.max_vertex().x, bb.max_vertex().y, bb.max_vertex().z);
            scene_diameter = std::max(scm::math::length(bb.max_vertex() - bb.min_vertex()), scene_diameter);
        }
        setInitialBound(bbox);
    }
    return 0;
}

void LamureDrawable::drawImplementation(osg::RenderInfo &renderInfo) const
{
    if (management_ == nullptr)
    {
        char *argv[] = { "opencover" };
        scm::shared_ptr<scm::core> scm_core(new scm::core(1, argv));

        int max_upload_budget = 64;
        int video_memory_budget = 2048;
        int main_memory_budget = 4096;


        lamure::ren::policy* policy = lamure::ren::policy::get_instance();
        policy->set_max_upload_budget_in_mb(max_upload_budget); //8
        policy->set_render_budget_in_mb(video_memory_budget); //2048
        policy->set_out_of_core_budget_in_mb(main_memory_budget); //4096, 8192
        policy->set_window_width(coVRConfig::instance()->windows[0].sx);
        policy->set_window_height(coVRConfig::instance()->windows[0].sy);


        lamure::ren::model_database* database = lamure::ren::model_database::get_instance();

        std::vector<scm::math::mat4d> parsed_views = std::vector<scm::math::mat4d>();


        LamureDrawable *ld = (LamureDrawable *)this;

        std::set<lamure::model_t> visible_set;
        std::set<lamure::model_t> invisible_set;
        snapshot_session_descriptor snap_descriptor;
        for (int i = 0; i < 50; i++)
        {
            ld->model_transformations.push_back(scm::math::mat4f::identity()); // todo add set transform to renderer
        }
        ld->management_ = new management(model_filenames, model_transformations, visible_set, invisible_set, snap_descriptor);
        //management_->interpolate_between_measurement_transforms(measurement_file_interpolation);
        //management_->set_interpolation_step_size(measurement_interpolation_stepsize);
        //management_->enable_culling(pvs_culling);
        osg::Vec4 bg_rgb = VRViewer::instance()->getBackgroundColor();
        ld->management_->forward_background_color(bg_rgb[0], bg_rgb[1], bg_rgb[2]);

        // PVS basic setup. If no path is given, runtime access to the PVS will always return true (visible).
       /* if (pvs_file_path != "")
        {
            std::string pvs_grid_file_path = pvs_file_path;
            pvs_grid_file_path.resize(pvs_grid_file_path.length() - 3);
            pvs_grid_file_path = pvs_grid_file_path + "grid";

            lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();
            pvs->load_pvs_from_file(pvs_grid_file_path, pvs_file_path, false);
        }*/
        
    }
    const unsigned ctx = renderInfo.getState()->getContextID();
    while (ctx >= contextState.size())
    {
        // this will delete the old renderer contextState.resize(ctx+1);
        ContextState *nc = new ContextState;
        contextState.push_back(nc);
    }
   // vvRenderer *&renderer = contextState[ctx]->renderer;

   

      /*  for (std::vector<vvRenderState::ParameterType>::iterator it = contextState[ctx]->parameterChanges.begin();
            it != contextState[ctx]->parameterChanges.end();
            ++it)
        {
            renderer->setParameter(*it, renderState.getParameter(*it));
        }
        contextState[ctx]->parameterChanges.clear();
        if (contextState[ctx]->applyTF)
        {
            renderer->updateTransferFunction();
            contextState[ctx]->applyTF = false;
        }*/

        osg::ref_ptr<osg::StateSet> currentState = new osg::StateSet;
        renderInfo.getState()->captureCurrentState(*currentState);
        renderInfo.getState()->pushStateSet(currentState.get());

        // Render here:
        management_->MainLoop();

        renderInfo.getState()->popStateSet();
}