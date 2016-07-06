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
    void preFrame();
    void expandBoundingSphere(osg::BoundingSphere &bs);

private:

    PointReader* m_reader;

    osg::ref_ptr<osg::Geode> m_geode;
    osg::ref_ptr<PointRayTracerDrawable> m_drawable;

    visionaray::aligned_vector<sphere_type>                                             m_points;
    visionaray::aligned_vector<color_type, 32>                                          m_colors;
    visionaray::aabb                                                                    m_bbox;
    visionaray::tiled_sched<host_ray_type>                                              m_scheduler;

    host_bvh_type                                   m_host_bvh;
    std::vector<host_bvh_type>                      m_host_bvh_vector;
    std::vector<bvh_ref>                            m_host_bvh_refs;

};


#endif //POINT_RAY_TRACER_PLUGIN_H
