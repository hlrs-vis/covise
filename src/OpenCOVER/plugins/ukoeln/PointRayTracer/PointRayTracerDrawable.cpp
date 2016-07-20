#include <GL/glew.h>
#include "PointRayTracerDrawable.h"
#include "PointRayTracerKernel.h"
#include <iostream>

#include <cover/coVRConfig.h>
#include <cover/coVRLighting.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/VRViewer.h>

using namespace visionaray;

visionaray::mat4 osg_cast(osg::Matrixd const &m)
{
    float arr[16];
    std::copy(m.ptr(), m.ptr() + 16, arr);
    return visionaray::mat4(arr);
}


struct PointRayTracerDrawable::Impl
{
    Impl()
#ifdef __CUDACC__
        // Initialize CUDA scheduler with block dimensions
        : scheduler(8, 8) // TODO: determine best
#else
        // Initialize CPU scheduler with #threads
        : scheduler(15)
#endif
    {
    }

    sched_type                      scheduler;
    render_target_type              rt;
    point_vector*                   points;
    color_vector*                   colors;
    host_bvh_type*                  host_bvh;

#ifdef __CUDACC__
    device_bvh_type                 device_bvh;
    thrust::device_vector<bvh_ref>  bvh_refs;
    device_color_vector             device_colors;
#else
    std::vector<bvh_ref>            bvh_refs;
#endif
};


PointRayTracerDrawable::PointRayTracerDrawable()
    : m_impl(new Impl)
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
    for (auto const &point : m_impl->host_bvh->primitives())
    {
        bounds = combine(bounds, point.center);
    }

    auto c = bounds.center();
    osg::BoundingSphere::vec_type center(c.x, c.y, c.z);
    osg::BoundingSphere::value_type radius = length(c - bounds.min);
    bs.set(center, radius);
}

void PointRayTracerDrawable::initData(host_bvh_type &bvh, point_vector &points, color_vector &colors)
{
    m_impl->points   = &points;
    m_impl->colors   = &colors;
    m_impl->host_bvh = &bvh;
#ifdef __CUDACC__
    std::cout << "Copy data to GPU...\n";

    // Copy data
    m_impl->device_colors = device_color_vector(*m_impl->colors);
    m_impl->device_bvh    = device_bvh_type(bvh);
    // Create refs
    m_impl->bvh_refs.push_back(m_impl->device_bvh.ref());

    std::cout << "Ready\n";
#else
    m_impl->bvh_refs.push_back(bvh.ref());
#endif
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

    //delay rendering until we actually have data
    if (m_impl->bvh_refs.size() == 0)
        return;

    auto vparams = getViewingParams(info);
    if(vparams.width != m_impl->rt.width() || vparams.height != m_impl->rt.height()){
        m_impl->rt.resize(vparams.width, vparams.height);
    }

    auto sparams = make_sched_params(
        vparams.view_matrix,
        vparams.proj_matrix,
        m_impl->rt);

    // some setup
    using R = ray_type;
    using S = R::scalar_type;
    using C = visionaray::vector<4, S>;

#ifdef __CUDACC__
    using B = decltype(thrust::raw_pointer_cast(m_impl->bvh_refs.data()));
    using CC = decltype(thrust::raw_pointer_cast(m_impl->device_colors.data()));

    Kernel<B, CC> kernel(
        thrust::raw_pointer_cast(m_impl->bvh_refs.data()),
        thrust::raw_pointer_cast(m_impl->bvh_refs.data()) + m_impl->bvh_refs.size(),
        thrust::raw_pointer_cast(m_impl->device_colors.data())
        );
#else
    using B = decltype(m_impl->bvh_refs.data());
    using CC = decltype(m_impl->colors->data());

    Kernel<B, CC> kernel(
        m_impl->bvh_refs.data(),
        m_impl->bvh_refs.data() + m_impl->bvh_refs.size(),
        m_impl->colors->data()
        );
#endif

    // call scheduler for actual rendering
    m_impl->scheduler.frame(kernel, sparams);

    //display result
    sparams.rt.display_color_buffer();

    //update member
    m_total_frame_num++;
}



