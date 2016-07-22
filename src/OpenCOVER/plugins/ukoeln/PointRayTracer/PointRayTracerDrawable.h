#ifndef POINT_RAY_TRACER_DRAWABLE_H
#define POINT_RAY_TRACER_DRAWABLE_H

#include <memory>

#include <osg/Drawable>

#include "PointRayTracerGlobals.h"

class PointRayTracerDrawable : public osg::Drawable
{
public:
    PointRayTracerDrawable();
    ~PointRayTracerDrawable();

    void expandBoundingSphere(osg::BoundingSphere &bs);

    void initData(host_bvh_type &bvh, point_vector &points, color_vector &colors);

private:

    //osg::Drawable interface
    PointRayTracerDrawable* cloneType() const;    
    osg::Object* clone(const osg::CopyOp& op) const;
    PointRayTracerDrawable(PointRayTracerDrawable const& rhs, osg::CopyOp const& op = osg::CopyOp::SHALLOW_COPY);
    void drawImplementation(osg::RenderInfo& info) const;

    viewing_params getViewingParams(const osg::RenderInfo &info) const;

    //Use private implementation so that arch-specific members
    //aren't compiled differently by CUDA host and device compiler
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    mutable size_t             m_total_frame_num = 0;
    mutable bool               m_glewIsInitialized = false;

};


#endif //POINT_RAY_TRACER_DRAWABLE_H
