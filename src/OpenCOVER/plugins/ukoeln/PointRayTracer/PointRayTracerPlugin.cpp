 
/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include <GL/glew.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include "PointRayTracerPlugin.h"

using namespace osg;
PointRayTracerPlugin *PointRayTracerPlugin::plugin = NULL;

//-----------------------------------------------------------------------------

PointRayTracerPlugin::PointRayTracerPlugin() : m_scheduler(15)
{


}

bool PointRayTracerPlugin::init()
{
    if (cover->debugLevel(1)) fprintf(stderr, "\n    new PointRayTracerPlugin\n");

    //read config
    std::string filename = covise::coCoviseConfig::getEntry("COVER.Plugin.PointRayTracer.Filename");
    if(filename.empty()) filename = "/data/KleinAltendorf/ausschnitte/test_UTM_klein.pts";

    float pointSize = covise::coCoviseConfig::getFloat("COVER.Plugin.PointRayTracer.PointSize",0.01f);

    //create reader and read data into arrays
    m_reader = new PointReader();
    if(!m_reader->readFile(filename, pointSize, m_points, m_colors, m_bbox, true)) return false;

    //build bvh
    m_host_bvh = visionaray::build<host_bvh_type>(m_points.data(), m_points.size());
    m_host_bvh_refs.push_back(m_host_bvh.ref());

    //init drawable and set pointers to data
    m_drawable = new PointRayTracerDrawable;
    m_drawable->m_host_bvh_refs = &m_host_bvh_refs;
    m_drawable->m_points = &m_points;
    m_drawable->m_colors = &m_colors;
    m_drawable->m_scheduler = &m_scheduler;

    //init geode and add it to the scenegraph
    m_geode = new osg::Geode;
    m_geode->setName("PointRayTracer");
    m_geode->addDrawable(m_drawable);
    opencover::cover->getScene()->addChild(m_geode);

    return true;
}

PointRayTracerPlugin::~PointRayTracerPlugin()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    delete PointRayTracerPlugin\n");

    delete m_reader;

}

void PointRayTracerPlugin::preFrame()
{
//    if (cover->debugLevel(1)) fprintf(stderr, "\n    preFrame PointRayTracerPlugin\n");
}

void PointRayTracerPlugin::expandBoundingSphere(osg::BoundingSphere &bs)
{
    m_drawable->expandBoundingSphere(bs);
}

COVERPLUGIN(PointRayTracerPlugin)
