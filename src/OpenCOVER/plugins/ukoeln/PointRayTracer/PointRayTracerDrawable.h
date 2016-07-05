#ifndef POINT_RAY_TRACER_DRAWABLE_H
#define POINT_RAY_TRACER_DRAWABLE_H

#include <osg/Drawable>

#include "PointRayTracerGlobals.h"

class PointRayTracerDrawable : public osg::Drawable
{
public:
    PointRayTracerDrawable();
    ~PointRayTracerDrawable();

    void expandBoundingSphere(osg::BoundingSphere &bs);

    std::vector<bvh_ref>*                        m_host_bvh_refs;
    visionaray::aligned_vector<sphere_type>*     m_points;
    visionaray::aligned_vector<color_type, 32>*  m_colors;
    visionaray::tiled_sched<host_ray_type>*      m_scheduler;

private:

    //osg::Drawable interface
    PointRayTracerDrawable* cloneType() const;    
    osg::Object* clone(const osg::CopyOp& op) const;
    PointRayTracerDrawable(PointRayTracerDrawable const& rhs, osg::CopyOp const& op = osg::CopyOp::SHALLOW_COPY);
    void drawImplementation(osg::RenderInfo& info) const;

    viewing_params getViewingParams(const osg::RenderInfo &info) const;

    mutable size_t                  m_total_frame_num = 0;
    mutable host_render_target_type m_host_rt;
    mutable bool                    m_glewIsInitialized = false;

};


#endif //POINT_RAY_TRACER_DRAWABLE_H
