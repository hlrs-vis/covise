/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: TrackerMarker Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <osg/ShapeDrawable>
#include <osg/Shape>
#include <osg/Geode>
#include <device/VRTracker.h>
#include <device/coVRTrackingSystems.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRMSController.h>
#include "TrackerMarker.h"

TrackerMarkerPlugin::TrackerMarkerPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool TrackerMarkerPlugin::init()
{
    group = new osg::Group();
    cover->getScene()->addChild(group);

    return true;
}

// this is called if the plugin is removed at runtime
TrackerMarkerPlugin::~TrackerMarkerPlugin()
{
    cover->getScene()->removeChild(group);
}

void
TrackerMarkerPlugin::preFrame()
{
    int numMarkers = 0;
    float *pos = NULL;
    int *visible = NULL;
    coVRTrackingSystems *ph = VRTracker::instance()->getTrackingSystemsImpl();
    if (coVRMSController::instance()->isMaster())
    {
        if (ph)
            numMarkers = ph->getNumMarkers();
        coVRMSController::instance()->sendSlaves((char *)&numMarkers, sizeof(numMarkers));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&numMarkers, sizeof(numMarkers));
    }

    pos = new float[3 * numMarkers];
    visible = new int[numMarkers];

    if (coVRMSController::instance()->isMaster())
    {
        if (ph)
        {
            for (int i = 0; i < numMarkers; i++)
            {
                visible[i] = ph->getMarker(i, &pos[i * 3]);
            }
        }
        coVRMSController::instance()->sendSlaves((char *)pos, sizeof(float) * 3 * numMarkers);
        coVRMSController::instance()->sendSlaves((char *)visible, sizeof(int) * numMarkers);
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)pos, sizeof(float) * 3 * numMarkers);
        coVRMSController::instance()->readMaster((char *)visible, sizeof(int) * numMarkers);
    }

    osg::Geode *geode = new osg::Geode();

    for (int i = 0; i < numMarkers; i++)
    {
        if (visible[i])
        {
            osg::ShapeDrawable *draw = new osg::ShapeDrawable;
            draw->setShape(new osg::Sphere(osg::Vec3(pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]), 3.0));
            geode->addDrawable(draw);
        }
    }
    delete[] pos;
    delete[] visible;

    if (group->getNumChildren() > 10)
    {
        group->removeChild((unsigned)0);
    }
    group->addChild(geode);
}

COVERPLUGIN(TrackerMarkerPlugin)
