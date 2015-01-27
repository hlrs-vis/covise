/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRAxisProjection                                        **
 **              Draws the way to the point                                **
 **                following the x, y and z axis                           **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "coVRAxisProjection.h"

#include <cover/coVRPluginSupport.h>

#include <osg/ShapeDrawable>

using namespace osg;
using namespace opencover;

//
// Constructor
//
coVRAxisProjection::coVRAxisProjection(Vec3 point, double dashRadius, double dashLength)
    : point_(point)
    , dashRadius_(dashRadius)
    , dashLength_(dashLength)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRAxisProjection::coVRAxisProjection\n");

    // the node for the scenegraph
    node_ = new MatrixTransform();
    node_->ref();

    updateXProjection();
    updateYProjection();
    updateZProjection();

    cover->getObjectsRoot()->addChild(node_.get());
}

//
// Destructor
//
coVRAxisProjection::~coVRAxisProjection()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRAxisProjection::~coVRAxisProjection\n");

    cover->getObjectsRoot()->removeChild(node_.get());
    node_->unref();
}

//----------------------------------------------------------------------
void coVRAxisProjection::setVisible(bool visible)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRAxisProjection::setVisible %d\n", visible);

    if (visible)
        node_->setNodeMask(node_->getNodeMask() | (Isect::Visible));
    else
        node_->setNodeMask(node_->getNodeMask() & (~Isect::Visible));
}

//----------------------------------------------------------------------
void coVRAxisProjection::setPoint(Vec3 point)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRAxisProjection::setStartpoint\n");

    point_ = point;
}

void coVRAxisProjection::update(Vec3 point)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRAxisProjection::update (%f %f %f)\n", point.x(), point.y(), point.z());

    point_ = point;

    // first remove all children of geode
    int all = node_->getNumChildren();
    node_->removeChildren(0, all);

    updateXProjection();
    updateYProjection();
    updateZProjection();

    // renew node
    node_->dirtyBound();
}

//----------------------------------------------------------------------
void coVRAxisProjection::updateXProjection()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRAxisProjection::updateXProjection\n");

    if (point_.x() != 0.0)
    {
        int fac = 1;
        int end = (int)point_.x();
        double rest = point_.x() - end;

        //fprintf(stderr, "updateXProjection rest = %f end = %d\n", rest, end);

        if (point_.x() < 0)
        {
            fac = -1;
            // make positiv
            end *= fac;
            rest *= fac;
        }
        Geode *dashes = new Geode();

        // drawing the dashed line from the origin to end
        for (int i = 0; i < end; i++)
        {
            Cylinder *cylinder = new Cylinder();
            double x = fac * ((double)i + dashLength_ / 2.0 + 0.1);
            cylinder->set(Vec3(x, 0.0, 0.0), dashRadius_, dashLength_);
            Matrix rotation;
            rotation.makeRotate(inDegrees(90.0), 0.0, 1.0, 0.0);
            cylinder->setRotation(rotation.getRotate());
            ShapeDrawable *cylinderDraw = new ShapeDrawable();
            cylinderDraw->setShape(cylinder);
            dashes->addDrawable(cylinderDraw);
            cylinderDraw->setUseDisplayList(false);
        }

        if (rest > 0.1) // to small for drawing
        {
            // drawing the rest of the dashed line
            Cylinder *cylinder = new Cylinder();
            double x = fac * ((double)end + (rest - 0.1) / 2.0 + 0.1);
            cylinder->set(Vec3(x, 0.0, 0.0), dashRadius_, rest - 0.1);
            Matrix rotation;
            rotation.makeRotate(inDegrees(90.0), 0.0, 1.0, 0.0);
            cylinder->setRotation(rotation.getRotate());
            ShapeDrawable *cylinderDraw = new ShapeDrawable();
            cylinderDraw->setShape(cylinder);
            dashes->addDrawable(cylinderDraw);
            cylinderDraw->setUseDisplayList(false);
        }

        node_->addChild(dashes);
    }
}

//----------------------------------------------------------------------
void coVRAxisProjection::updateYProjection()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRAxisProjection::updateYProjection\n");

    if (point_.y() != 0.0)
    {
        int fac = 1;
        int end = (int)point_.y();
        double rest = point_.y() - end;
        if (point_.y() < 0)
        {
            fac = -1;
            // make positiv
            end *= fac;
            rest *= fac;
        }
        Geode *dashes = new Geode();

        // drawing the dashed line from the origin to end
        for (int i = 0; i < end; i++)
        {
            Cylinder *cylinder = new Cylinder();
            double y = fac * ((double)i + dashLength_ / 2.0 + 0.1);
            cylinder->set(Vec3(point_.x(), y, 0.0), dashRadius_, dashLength_);
            Matrix rotation;
            rotation.makeRotate(inDegrees(90.0), 1.0, 0.0, 0.0);
            cylinder->setRotation(rotation.getRotate());
            ShapeDrawable *cylinderDraw = new ShapeDrawable();
            cylinderDraw->setShape(cylinder);
            dashes->addDrawable(cylinderDraw);
            cylinderDraw->setUseDisplayList(false);
        }

        if (rest > 0.1) // to small for drawing
        {
            // drawing the rest of the dashed line
            Cylinder *cylinder = new Cylinder();
            double y = fac * ((double)end + (rest - 0.1) / 2.0 + 0.1);
            cylinder->set(Vec3(point_.x(), y, 0.0), dashRadius_, rest - 0.1);
            Matrix rotation;
            rotation.makeRotate(inDegrees(90.0), 1.0, 0.0, 0.0);
            cylinder->setRotation(rotation.getRotate());
            ShapeDrawable *cylinderDraw = new ShapeDrawable();
            cylinderDraw->setShape(cylinder);
            dashes->addDrawable(cylinderDraw);
            cylinderDraw->setUseDisplayList(false);
        }

        node_->addChild(dashes);
    }
}

//----------------------------------------------------------------------
void coVRAxisProjection::updateZProjection()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRAxisProjection::updateZProjection\n");

    if (point_.z() != 0.0)
    {
        int fac = 1;
        int end = (int)point_.z();
        double rest = point_.z() - end;
        if (point_.z() < 0)
        {
            fac = -1;
            // make positiv
            end *= fac;
            rest *= fac;
        }
        Geode *dashes = new Geode();

        // drawing the dashed line from the origin to end
        for (int i = 0; i < end; i++)
        {
            Cylinder *cylinder = new Cylinder();
            double z = fac * ((double)i + dashLength_ / 2.0 + 0.1);
            cylinder->set(Vec3(point_.x(), point_.y(), z), dashRadius_, dashLength_);
            ShapeDrawable *cylinderDraw = new ShapeDrawable();
            cylinderDraw->setShape(cylinder);
            dashes->addDrawable(cylinderDraw);
            cylinderDraw->setUseDisplayList(false);
        }

        if (rest > 0.1) // to small for drawing
        {
            // drawing the rest of the dashed line
            Cylinder *cylinder = new Cylinder();
            double z = fac * ((double)end + (rest - 0.1) / 2.0 + 0.1);
            cylinder->set(Vec3(point_.x(), point_.y(), z), dashRadius_, rest - 0.1);
            ShapeDrawable *cylinderDraw = new ShapeDrawable();
            cylinderDraw->setShape(cylinder);
            dashes->addDrawable(cylinderDraw);
            cylinderDraw->setUseDisplayList(false);
        }

        node_->addChild(dashes);
    }
}
