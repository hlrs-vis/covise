/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <EarthViewpointManager.h>
#include "EarthPlugin.h"

#include <osgEarth/Map>
#include <osgEarth/Viewpoint>
#include <osgEarthAnnotation/AnnotationNode>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/osg/mathUtils.h>

#include <osg/MatrixTransform>

using namespace osgEarth;
using namespace opencover;

EarthViewpointManager::EarthViewpointManager(coTUITab *et)
    : earthTab(et)
{
    ViewpointsFrame = new coTUIFrame("Viewpoints", et->getID());
    ViewpointsFrame->setPos(1, 1);
    ViewpointOptionsFrame = new coTUIFrame("ViewpointOptions", et->getID());
    ViewpointOptionsFrame->setPos(1, 2);

    printButton = new coTUIButton("Print", ViewpointOptionsFrame->getID());
    printButton->setEventListener(this);
    printButton->setPos(0, 0);
    rpmLabel = new coTUILabel("rpm", ViewpointOptionsFrame->getID());
    rpmLabel->setEventListener(this);
    rpmLabel->setPos(0, 1);
    rpmField = new coTUIEditFloatField("xf", ViewpointOptionsFrame->getID(), false);
    rpmField->setPos(1, 1);
    rpmField->setEventListener(this);
    rpmField->setValue(1);
}

void EarthViewpointManager::addViewpoint(Viewpoint *vp)
{
    EarthViewpoint *v = new EarthViewpoint(ViewpointsFrame, vp, viewpoints.size());
    viewpoints.push_back(v);
    if (viewpoints.size() == 1)
        v->setViewpoint();
}

void EarthViewpointManager::tabletPressEvent(coTUIElement *e)
{
    if (e == printButton)
    {
        osg::Vec3d currentPos;
        osg::Vec3d currentLatLong;
        osg::Matrix xformMat = cover->getXformMat();
        osg::Matrix rotMat = xformMat;
        rotMat.setTrans(0, 0, 0);
        osg::Matrix invRot;
        invRot.invert(rotMat);
        osg::Matrix xformOnlyMat = xformMat * invRot;
        currentPos = xformOnlyMat.getTrans();
        currentPos /= -cover->getScale();

        if (const SpatialReference *srs = EarthPlugin::plugin->getSRS())
        {
            //srs->transformFromECEF(currentPos,currentLatLong);
            srs->transformFromWorld(currentPos, currentLatLong);

            // convert to geocentric coords if necessary:
            if (EarthPlugin::plugin->getMapNode()->getMap()->isGeocentric())
            {
                osg::Matrix toUpright;

                toUpright.makeRotate(currentLatLong.x() / 180.0 * M_PI, osg::X_AXIS, -(90.0 - currentLatLong.y()) / 180.0 * M_PI, osg::Y_AXIS, 0 / 180.0 * M_PI, osg::Z_AXIS);
                osg::Matrix invToUpright;
                invToUpright.invert(toUpright);
                rotMat = invToUpright * rotMat;
            }
            //TODO remove roll
            float pitch, heading;
            osg::Vec3 zaxis(rotMat(2, 0), rotMat(2, 1), rotMat(2, 2));
            osg::Vec3 xaxis(rotMat(0, 0), rotMat(0, 1), rotMat(0, 2));

            pitch = acos(osg::Z_AXIS * zaxis);
            heading = acos(osg::X_AXIS * xaxis);
            if (xaxis[1] < 0)
                heading = 2 * M_PI - heading;
            heading = -osg::RadiansToDegrees(heading) + 360.0;
            pitch = -osg::RadiansToDegrees(pitch);

            char name[1000];
            sprintf(name, "NewViewpoint %d[%f]", (int)viewpoints.size(), cover->getScale());
            fprintf(stdout, "<viewpoint name=\"NewViewpoint %d[%f]\" lat=\"%lf\" long=\"%lf\" height=\"%lf\" heading=\"%lf\" pitch=\"%lf\" range=\"0\"/>\n", (int)viewpoints.size(), cover->getScale(), currentLatLong.y(), currentLatLong.x(), currentLatLong.z(), heading, pitch);
            addViewpoint(new Viewpoint(name, currentLatLong[0],currentLatLong[1],currentLatLong[2],heading, pitch, 0));
        }
    }
}
