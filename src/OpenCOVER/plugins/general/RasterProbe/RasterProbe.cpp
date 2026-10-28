/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * Author: Siva Karthikeya Mandarapu
 * License: LGPL 2+ */
#include "RasterProbe.h"

#include <algorithm>

#include <gdal.h>
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/input/input.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <geodata/GeoData.h>
#include <vrml97/vrml/VrmlNamespace.h>

#include "VrmlNodeRasterProbe.h"

using namespace opencover;

std::vector<VrmlNodeRasterProbe *> RasterProbe::s_probes;

RasterProbe::RasterProbe()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("RasterProbe", cover->ui)
{
    GDALAllRegister();
    vrml::VrmlNamespace::addBuiltIn(vrml::VrmlNode::defineType<VrmlNodeRasterProbe>());
}

bool RasterProbe::init()
{
    auto menu = new ui::Menu("RasterProbe", this);
    m_probeButton = new ui::Button(menu, "Probe");
    m_probeButton->setText("Show values under pointer");
    m_probeButton->setState(true);
    return true;
}

void RasterProbe::addProbe(VrmlNodeRasterProbe *probe)
{
    s_probes.push_back(probe);
}

void RasterProbe::removeProbe(VrmlNodeRasterProbe *probe)
{
    s_probes.erase(std::remove(s_probes.begin(), s_probes.end(), probe), s_probes.end());
}

void RasterProbe::preFrame()
{
    // only the probe enabled last sends, so several can share one display
    VrmlNodeRasterProbe *probe = nullptr;
    for (auto *p : s_probes)
    {
        if (p->enabled() && (!probe || p->enableSerial() > probe->enableSerial()))
            probe = p;
    }

    // only the master samples; syncData must run every frame on every node
    struct
    {
        double value = 0.0;
        int valid = 0;
    } sample;
    if (coVRMSController::instance()->isMaster() && m_probeButton->state() && probe)
    {
        // own ray, cover->getIntersectedNode() misses the terrain overlays
        const osg::Matrix &pointer = Input::instance()->isTrackingOn() ? cover->getPointerMat() : cover->getMouseMat();
        osg::Vec3d start = osg::Vec3d(0, 0, 0) * pointer * cover->getInvBaseMat();
        osg::Vec3d end = osg::Vec3d(0, 1e9, 0) * pointer * cover->getInvBaseMat();

        osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector = new osgUtil::LineSegmentIntersector(start, end);
        intersector->setIntersectionLimit(osgUtil::Intersector::LIMIT_NEAREST);
        osgUtil::IntersectionVisitor visitor(intersector.get());
        cover->getObjectsRoot()->accept(visitor);

        if (intersector->containsIntersections())
        {
            osg::Vec3d pos = intersector->getFirstIntersection().getWorldIntersectPoint() * GeoData::instance()->inverseProjectTransform();
            sample.valid = probe->sample(pos.x(), pos.y(), sample.value);
        }
    }
    coVRMSController::instance()->syncData(&sample, sizeof(sample));

    if (probe)
        probe->setValue(sample.valid, sample.value, probe != m_lastProbe); // resend when another probe took over
    m_lastProbe = probe;
}

COVERPLUGIN(RasterProbe)
