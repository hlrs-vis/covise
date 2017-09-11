/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <EarthViewpoint.h>
#include "EarthPlugin.h"

#include <osgEarth/Map>
#include <osgEarth/Version>
#include <osgEarth/Viewpoint>
#include <osgEarthAnnotation/AnnotationNode>

#include <cover/coVRPluginSupport.h>

#include <osg/MatrixTransform>

using namespace osgEarth;

EarthViewpoint::EarthViewpoint(coTUIFrame *parent, Viewpoint *vp, int ViewpointNumber)
{
    parentFrame = parent;
    viewpoint = vp;
    viewpointNumber = ViewpointNumber;
#if OSGEARTH_VERSION_GREATER_THAN(2,6,0)
    viewpointButton = new coTUIButton(viewpoint->name().get(), parentFrame->getID());
#else
    viewpointButton = new coTUIButton(viewpoint->getName(), parentFrame->getID());
#endif
    viewpointButton->setEventListener(this);
    viewpointButton->setPos(0, ViewpointNumber);
    animateButton = new coTUIToggleButton("animate", parentFrame->getID());
    animateButton->setEventListener(this);
    animateButton->setPos(1, ViewpointNumber);
    if (cover != NULL)
        cover->getUpdateManager()->add(this);
}

void EarthViewpoint::setScale()
{
#if OSGEARTH_VERSION_GREATER_THAN(2,6,0)
    const char *name = viewpoint->name().get().c_str();
#else
    const char *name = viewpoint->getName().c_str();
#endif
    const char *nameEnd = name + strlen(name);
    while (nameEnd > name)
    {
        if (*nameEnd == '[')
        {
            float scale = 0;
            sscanf(nameEnd + 1, "%f", &scale);
            if (scale != 0)
            {
                cover->setScale(scale);
            }
            break;
        }
        nameEnd--;
    }
}
void EarthViewpoint::computeToUpright()
{
#if OSGEARTH_VERSION_GREATER_THAN(2,6,0)
    new_center = viewpoint->focalPoint().get().vec3d();
#else
    new_center = viewpoint->getFocalPoint();
#endif
    osg::Vec3d localUpVector = osg::Z_AXIS;
    osg::Vec3d upVector = osg::Z_AXIS;
    toUpright.makeIdentity();
    // start by transforming the requested focal point into world coordinates:
    /*if (const SpatialReference *srs = EarthPlugin::plugin->getSRS())
    {
        // resolve the VP's srs. If the VP's SRS is not specified, assume that it
        // is either lat/long (if the map is geocentric) or X/Y (otherwise).
        osg::ref_ptr<const SpatialReference> vp_srs = viewpoint->SRS() ? viewpointgetSRS() : EarthPlugin::plugin->getMapNode()->getMap()->isGeocentric() ? srs->getGeographicSRS() : srs;

        if (!srs->isEquivalentTo(vp_srs.get()))
        {
            osg::Vec3d local = new_center;
            // reproject the focal point if necessary:
            vp_srs->transform2D(new_center.x(), new_center.y(), srs, local.x(), local.y());
            new_center = local;
        }

        // convert to geocentric coords if necessary:
        if (EarthPlugin::plugin->getMapNode()->getMap()->isGeocentric())
        {
            osg::Vec3d geocentric;

            srs->getEllipsoid()->convertLatLongHeightToXYZ(
                osg::DegreesToRadians(new_center.y()),
                osg::DegreesToRadians(new_center.x()),
                new_center.z(),
                geocentric.x(), geocentric.y(), geocentric.z());

            toUpright.makeRotate(new_center.x() / 180.0 * M_PI, osg::X_AXIS, -(90.0 - new_center.y()) / 180.0 * M_PI, osg::Y_AXIS, 0 / 180.0 * M_PI, osg::Z_AXIS);
            new_center = geocentric;
            localUpVector = srs->getEllipsoid()->computeLocalUpVector(osg::DegreesToRadians(new_center.y()), osg::DegreesToRadians(new_center.x()), new_center.z());
        }
    }*/
}

void EarthViewpoint::setViewpoint()
{
    setScale();
    computeToUpright();
    cover->setXformMat(osg::Matrix::translate(-new_center * cover->getScale()) * toUpright * osg::Matrix::rotate(osg::DegreesToRadians(-viewpoint->getHeading()), osg::Z_AXIS) * osg::Matrix::rotate(osg::DegreesToRadians(-viewpoint->getPitch()), osg::X_AXIS) * osg::Matrix::translate(0, viewpoint->getRange() * cover->getScale(), 0));
}
void EarthViewpoint::tabletPressEvent(coTUIElement *e)
{
    if (e == viewpointButton)
    {
        setViewpoint();
    }
}

bool EarthViewpoint::update()
{
    if (animateButton->getState())
    {
        if (firstTime)
        {
            setScale();
            computeToUpright();
            startTime = cover->frameTime();
            firstTime = false;
        }
        cover->setXformMat(osg::Matrix::translate(-new_center * cover->getScale()) * toUpright * osg::Matrix::rotate(osg::DegreesToRadians(-viewpoint->getHeading() + ((cover->frameTime() - startTime) / 60.0) * 360.0 * EarthPlugin::plugin->getRPM()), osg::Z_AXIS) * osg::Matrix::rotate(osg::DegreesToRadians(-viewpoint->getPitch()), osg::X_AXIS) * osg::Matrix::translate(0, viewpoint->getRange() * cover->getScale(), 0));
    }
    else
        firstTime = true;

    return true;
}
