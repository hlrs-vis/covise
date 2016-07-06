#include <GL/glew.h>
#include "PointRayTracerDrawable.h"
#include <iostream>

#include <cover/coVRConfig.h>
#include <cover/coVRLighting.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/VRViewer.h>

#include <visionaray/get_color.h>

using namespace visionaray;

visionaray::mat4 osg_cast(osg::Matrixd const &m)
{
    float arr[16];
    std::copy(m.ptr(), m.ptr() + 16, arr);
    return visionaray::mat4(arr);
}


PointRayTracerDrawable::PointRayTracerDrawable()
{
    setSupportsDisplayList(false);
}

PointRayTracerDrawable::~PointRayTracerDrawable()
{

}

viewing_params PointRayTracerDrawable::getViewingParams(const osg::RenderInfo& info) const
{
    //get stereo mode, default to right eye
    osg::DisplaySettings::StereoMode stereoMode = osg::DisplaySettings::StereoMode::RIGHT_EYE;
    if(auto state = info.getState()) {
        if(auto ds = state->getDisplaySettings()) {
            stereoMode = ds->getStereoMode();
        }
    }

    //Left or Right Eye?
    eye current_eye = Right; //default
    if(opencover::coVRConfig::instance()->stereoState())
    {
        switch(stereoMode)
        {
            case osg::DisplaySettings::ANAGLYPHIC:
            {
                if(m_total_frame_num % 2 == 0){
                    current_eye = Left;
                }
            }
            break;

            case osg::DisplaySettings::QUAD_BUFFER:
            {
                GLint db = 0;
                glGetIntegerv(GL_DRAW_BUFFER, &db);
                if(db != GL_BACK_RIGHT && db != GL_FRONT_RIGHT && db != GL_RIGHT){
                    current_eye = Left;
                }
            }
            break;

            case osg::DisplaySettings::LEFT_EYE:
            {
                current_eye = Left;
            }
            break;

            case osg::DisplaySettings::RIGHT_EYE:
            {
                current_eye = Right;
            }
            break;

            default:
                break;
        }
    }

    // Matrices
    auto t = opencover::cover->getXformMat();
    auto s = opencover::cover->getObjectsScale()->getMatrix();
    auto v = current_eye == Right
                 ? opencover::coVRConfig::instance()->channels[0].rightView
                 : opencover::coVRConfig::instance()->channels[0].leftView;
    auto view = osg_cast(s * t * v);
    auto proj = current_eye == Right
                    ? osg_cast(opencover::coVRConfig::instance()->channels[0].rightProj)
                    : osg_cast(opencover::coVRConfig::instance()->channels[0].leftProj);

    // Viewport
    auto osg_cam = opencover::coVRConfig::instance()->channels[0].camera;
    auto osg_viewport = osg_cam->getViewport();
    int w = osg_viewport->width();
    int h = osg_viewport->height();

    viewing_params retval;
    retval.view_matrix = view;
    retval.proj_matrix = proj;
    retval.width = w;
    retval.height = h;

    return retval;
}

void PointRayTracerDrawable::expandBoundingSphere(osg::BoundingSphere &bs)
{
    aabb bounds(vec3(std::numeric_limits<float>::max()), -vec3(std::numeric_limits<float>::max()));
    for (auto const &point : *m_points)
    {
        bounds = combine(bounds, point.center);
    }

    auto c = bounds.center();
    osg::BoundingSphere::vec_type center(c.x, c.y, c.z);
    osg::BoundingSphere::value_type radius = length(c - bounds.min);
    bs.set(center, radius);
}

PointRayTracerDrawable::PointRayTracerDrawable(const PointRayTracerDrawable &rhs, const osg::CopyOp &op)
{
    setSupportsDisplayList(false);
}

PointRayTracerDrawable *PointRayTracerDrawable::cloneType() const
{
    return new PointRayTracerDrawable();
}

osg::Object* PointRayTracerDrawable::clone(const osg::CopyOp &op) const
{
    return new PointRayTracerDrawable(*this, op);
}

void PointRayTracerDrawable::drawImplementation(osg::RenderInfo &info) const
{
    //init glew on first draw call
    if(!m_glewIsInitialized){
        if(glewInit() == GLEW_OK) m_glewIsInitialized = true;
    }
    //quit if init glew did not work
    if(!m_glewIsInitialized) return;

    auto vparams = getViewingParams(info);
    if(vparams.width != m_host_rt.width() || vparams.height != m_host_rt.height()){
        m_host_rt.resize(vparams.width, vparams.height);
    }

    auto sparams = make_sched_params(
        vparams.view_matrix,
        vparams.proj_matrix,
        m_host_rt);

    // some setup
    using R = host_ray_type;
    using S = R::scalar_type;
    using C = visionaray::vector<4, S>;

    auto bgcolor = visionaray::vec3(0.2,0.2,0.2);

    // kernel with ray tracing logic
    auto kernel = [&](R ray) -> visionaray::result_record<S>
    {
        visionaray::result_record<S> result;
        result.color = C(bgcolor, 1.0f);

        auto hit_rec = visionaray::closest_hit(
                ray,
                m_host_bvh_refs->begin(),
                m_host_bvh_refs->end()
                );

        result.hit = hit_rec.hit;
        result.isect_pos = ray.ori + ray.dir * hit_rec.t;

        auto color = get_color(m_colors->data(),hit_rec,host_bvh_type(),visionaray::colors_per_face_binding());

        result.color = select(
                hit_rec.hit,
                C(color, S(1.0)),
                result.color
                );

        return result;
    };

    // call scheduler for actual rendering
    m_scheduler->frame(kernel, sparams);

    //display result
    sparams.rt.display_color_buffer();

    //update member
    m_total_frame_num++;
}



