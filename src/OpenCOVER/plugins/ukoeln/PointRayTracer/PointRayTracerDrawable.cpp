#include "ColorSphere.h"
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
    std::vector<host_bvh_type>*     host_bvh_vector;

#ifdef __CUDACC__
    thrust::device_vector<bvh_ref>  bvh_refs;
    std::vector<device_bvh_type>    device_bvh_vector;
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

    for(int i = 0; i < m_impl->host_bvh_vector->size(); i++){

        if (m_impl->host_bvh_vector->at(i).num_nodes() > 0)
        {
            bounds = combine(bounds, m_impl->host_bvh_vector->at(i).node(0).get_bounds());
        }
    }

    auto c = bounds.center();
    osg::BoundingSphere::vec_type center(c.x, c.y, c.z);
    osg::BoundingSphere::value_type radius = length(c - bounds.min);
    bs.set(center, radius);

}

void PointRayTracerDrawable::initData(std::vector<host_bvh_type> &bvh_vector)
{
    m_impl->host_bvh_vector = &bvh_vector;

#ifdef __CUDACC__

    std::cout << "Copy data to GPU...\n";

    // Copy data
    m_impl->device_bvh_vector.resize(bvh_vector.size());
    for(size_t i = 0; i < bvh_vector.size(); i++){
        m_impl->device_bvh_vector[i] = device_bvh_type(bvh_vector[i]);
    }

    // Create refs
    m_impl->bvh_refs.push_back(m_impl->device_bvh_vector[0].ref());

#else    

    // Create refs
    m_impl->bvh_refs.push_back(bvh_vector[0].ref());

#endif
    std::cout << "Ready\n";
}

/*
void PointRayTracerDrawable::setVisibility(std::vector<bool>& visibility){

    if(m_impl->host_bvh_vector->size() != visibility.size()){
        std::cout << "PointRayTracerDrawable::setVisibility() ERROR: wrong number of visibility infos" << std::endl;
        return;
    }

    m_impl->bvh_refs.clear();

#ifdef __CUDACC__
    for(size_t i = 0; i < m_impl->device_bvh_vector.size(); i++){
        if(visibility[i]){
            m_impl->bvh_refs.push_back(m_impl->device_bvh_vector[i].ref());
        }
    }
#else
    for(size_t i = 0; i < m_impl->host_bvh_vector->size(); i++){
        if(visibility[i]){
            m_impl->bvh_refs.push_back(m_impl->host_bvh_vector->at(i).ref());
        }
    }
#endif

}
*/

void PointRayTracerDrawable::setCurrentPointCloud(int pointCloudID){
    if(pointCloudID < 0){
        std::cout << "PointRayTracerDrawable::setCurrentPointCloud pointCloud id < 0 : " << pointCloudID << std::endl;
        return;
     } else if (pointCloudID >= m_impl->host_bvh_vector->size()) {
        std::cout << "PointRayTracerDrawable::setCurrentPointCloud pointCloud too big. pointCloudID: " << pointCloudID << "  host_bvh_vector->size(): " << m_impl->host_bvh_vector->size() << std::endl;
        return;
    }

    m_impl->bvh_refs.clear();

#ifdef __CUDACC__
    m_impl->bvh_refs.push_back(m_impl->device_bvh_vector[pointCloudID].ref());
#else
    m_impl->bvh_refs.push_back(m_impl->host_bvh_vector->at(pointCloudID).ref());
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

    Kernel<B> kernel(
        thrust::raw_pointer_cast(m_impl->bvh_refs.data()),
        thrust::raw_pointer_cast(m_impl->bvh_refs.data()) + m_impl->bvh_refs.size()
        );
#else
    using B = decltype(m_impl->bvh_refs.data());    

    Kernel<B> kernel(
        m_impl->bvh_refs.data(),
        m_impl->bvh_refs.data() + m_impl->bvh_refs.size()
        );
#endif

    // call scheduler for actual rendering
    m_impl->scheduler.frame(kernel, sparams);

    //display result
    sparams.rt.display_color_buffer();

    //update member
    m_total_frame_num++;
}



