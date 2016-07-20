/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef POINT_RAY_TRACER_PLUGIN_H
#define POINT_RAY_TRACER_PLUGIN_H

#include "PointRayTracerGlobals.h"
#include "PointReader.h"
#include "PointRayTracerDrawable.h"

#include <cover/coVRPlugin.h>

using namespace opencover;



class PointRayTracerPlugin : public coVRPlugin
{

public:
    static PointRayTracerPlugin *plugin;

    PointRayTracerPlugin();
    virtual ~PointRayTracerPlugin();
    bool init();
    void preDraw(osg::RenderInfo &info);
    void expandBoundingSphere(osg::BoundingSphere &bs);

private:

    PointReader* m_reader;

    osg::ref_ptr<osg::Geode> m_geode;
    osg::ref_ptr<PointRayTracerDrawable> m_drawable;

    point_vector                                    m_points;
    color_vector                                    m_colors;
    visionaray::aabb                                m_bbox;

    host_bvh_type                                   m_host_bvh;
};


#endif //POINT_RAY_TRACER_PLUGIN_H
