/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRCoordinateAxis                                        **
 **              Draws the three coordinates axis                          **
**                 only positiv or both directions                         **
 **                with colors red (x), green (y), blue (z)                **
 **                and ticks per unit length                               **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "coVRCoordinateAxis.h"

#include <cover/coVRPluginSupport.h>

#include <osg/ShapeDrawable>
#include <osg/Material>

namespace osg
{
class Cylinder;
};

using namespace osg;
using namespace opencover;

//
// Constructor
//
coVRCoordinateAxis::coVRCoordinateAxis(double axisRadius, double axisLength, bool negAndPos, bool showTicks)
    : axisRadius_(axisRadius)
    , axisLength_(axisLength)
    , tickRadius_(axisRadius_ * 3.0)
    , tickLength_(axisRadius_ * 2.0)
    , negAndPos_(negAndPos)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRCoordinateAxis::coVRCoordinateAxis\n");

    // the node for the scenegraph
    axisNode_ = new MatrixTransform();
    axisNode_->ref();
    axisNode_->setName("CoordinateAxis");

    //fprintf(stderr,"axisRadius_ = %f  axisLength_ = %f axisFactor_ = %f;\n", axisRadius_, axisLength_, axisFactor_);
    makeAxis();
    if (showTicks)
        makeTicks();

    //do not intersect node
    axisNode_->setNodeMask(axisNode_->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick) & (Isect::Visible));

    // add to scenegraph
    cover->getObjectsRoot()->addChild(axisNode_.get());
}

//
// Destructor
//
coVRCoordinateAxis::~coVRCoordinateAxis()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "coVRCoordinateAxis::~coVRCoordinateAxis\n");
    cover->getObjectsRoot()->removeChild(axisNode_.get());
    axisNode_->unref();

    xTransform_->unref();
    yTransform_->unref();
    zTransform_->unref();

    xAxis_->unref();
    yAxis_->unref();
    zAxis_->unref();
}

//----------------------------------------------------------------------
void coVRCoordinateAxis::setVisible(bool visible)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCoordinateAxis::setVisible %d\n", visible);

    if (visible)
        axisNode_->setNodeMask(axisNode_->getNodeMask() | (Isect::Visible));
    else
        axisNode_->setNodeMask(axisNode_->getNodeMask() & (~Isect::Visible));
}

//----------------------------------------------------------------------
void coVRCoordinateAxis::makeAxis()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCoordinateAxis::makeAxis\n");

    Matrix m;

    // x axis - red
    xAxis_ = new coArrow(0.0, 0.0, false, false);
    xAxis_->ref();
    if (negAndPos_)
        xAxis_->drawArrow(Vec3(0.0, 0.0, 0.0), axisRadius_, 2.0 * axisLength_);
    else
        xAxis_->drawArrow(Vec3(0.0, 0.0, 0.0), axisRadius_, axisLength_);
    xAxis_->setColor(Vec4(1.0, 0.0, 0.0, 1.0)); // red
    xAxis_->setAmbient(Vec4(0.3, 0.0, 0.0, 1.0));
    xTransform_ = new MatrixTransform();
    xTransform_->ref();
    m.makeRotate(inDegrees(90.0), 0.0, 1.0, 0.0);
    if (negAndPos_)
        m.setTrans(-axisLength_, 0.0, 0.0);
    xTransform_->setMatrix(m);
    xTransform_->addChild(xAxis_.get());
    axisNode_->addChild(xTransform_.get());

    // y axis - green
    yAxis_ = new coArrow(0.0, 0.0, false, false);
    yAxis_->ref();
    if (negAndPos_)
        yAxis_->drawArrow(Vec3(0.0, 0.0, 0.0), axisRadius_, 2.0 * axisLength_);
    else
        yAxis_->drawArrow(Vec3(0.0, 0.0, 0.0), axisRadius_, axisLength_);
    yAxis_->setColor(Vec4(0.0, 1.0, 0.0, 1.0)); // green
    yAxis_->setAmbient(Vec4(0.0, 0.3, 0.0, 1.0));
    yTransform_ = new MatrixTransform();
    yTransform_->ref();
    m.makeRotate(inDegrees(270.0), 1.0, 0.0, 0.0);
    if (negAndPos_)
        m.setTrans(0.0, -axisLength_, 0.0);
    yTransform_->setMatrix(m);
    yTransform_->addChild(yAxis_.get());
    axisNode_->addChild(yTransform_.get());

    // z axis - blue
    zAxis_ = new coArrow(0.0, 0.0, false, false);
    zAxis_->ref();
    if (negAndPos_)
        zAxis_->drawArrow(Vec3(0.0, 0.0, 0.0), axisRadius_, 2.0 * axisLength_);
    else
        zAxis_->drawArrow(Vec3(0.0, 0.0, 0.0), axisRadius_, axisLength_);
    zAxis_->setColor(Vec4(0.0, 0.0, 1.0, 1.0)); //blue
    zAxis_->setAmbient(Vec4(0.0, 0.0, 0.3, 1.0));
    zTransform_ = new MatrixTransform();
    zTransform_->ref();
    if (negAndPos_)
    {
        m.makeTranslate(0.0, 0.0, -axisLength_);
        zTransform_->setMatrix(m);
    }
    zTransform_->addChild(zAxis_.get());
    axisNode_->addChild(zTransform_.get());
}

//----------------------------------------------------------------------
void coVRCoordinateAxis::makeTicks()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCoordinateAxis::makeTicks\n");

    // drawing ticks per unit for the axis like this: ----|----|----|----|----|----|----

    Geode *ticks = new Geode();

    int start;
    int end;

    if (negAndPos_)
    {
        start = -((int)(axisLength_)-1);
        end = (int)(axisLength_);
    }
    else
    {
        start = 1;
        end = (int)(axisLength_);
    }
    //fprintf(stderr,"start = %d   end = %d  axisLength_ = %f \n",start, end, axisLength_ );

    for (int i = start; i < end; i++)
    {
        // dont draw a tick into the origin
        if (i == 0)
            continue;

        Matrix rotationMatrix;

        // ticks for x axis
        Cylinder *xCylinder = new Cylinder();
        xCylinder->set(Vec3((double)i, 0.0, 0.0), tickRadius_, tickLength_);
        rotationMatrix.makeRotate(inDegrees(90.0), 0.0, 1.0, 0.0);
        xCylinder->setRotation(rotationMatrix.getRotate());
        ShapeDrawable *xCylinderDraw = new ShapeDrawable();
        xCylinderDraw->setShape(xCylinder);
        ticks->addDrawable(xCylinderDraw);
        StateSet *redStateSet = makeRedStateSet();
        xCylinderDraw->setStateSet(redStateSet);
        xCylinderDraw->setUseDisplayList(false);

        // ticks for y axis
        Cylinder *yCylinder = new Cylinder();
        yCylinder->set(Vec3(0.0, (double)i, 0.0), tickRadius_, tickLength_);
        rotationMatrix.makeRotate(inDegrees(270.0), 1.0, 0.0, 0.0);
        yCylinder->setRotation(rotationMatrix.getRotate());
        ShapeDrawable *yCylinderDraw = new ShapeDrawable();
        yCylinderDraw->setShape(yCylinder);
        ticks->addDrawable(yCylinderDraw);
        StateSet *greenStateSet = makeGreenStateSet();
        yCylinderDraw->setStateSet(greenStateSet);
        yCylinderDraw->setUseDisplayList(false);

        // ticks for z axis
        Cylinder *zCylinder = new Cylinder();
        zCylinder->set(Vec3(0.0, 0.0, (double)i), tickRadius_, tickLength_);
        ShapeDrawable *zCylinderDraw = new ShapeDrawable();
        zCylinderDraw->setShape(zCylinder);
        ticks->addDrawable(zCylinderDraw);
        StateSet *blueStateSet = makeBlueStateSet();
        zCylinderDraw->setStateSet(blueStateSet);
        zCylinderDraw->setUseDisplayList(false);
    }
    axisNode_->addChild(ticks);
}

//----------------------------------------------------------------------
StateSet *coVRCoordinateAxis::makeRedStateSet()
{
    Material *material = new Material();

    material->setColorMode(Material::OFF);
    material->setAmbient(Material::FRONT_AND_BACK, Vec4(1.0, 0.0, 0.0, 1.0));
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.2, 0.0, 0.0, 1.0));
    material->setAlpha(Material::FRONT_AND_BACK, 1.0);

    StateSet *stateSet = new StateSet();
    stateSet->setRenderingHint(StateSet::OPAQUE_BIN);
    stateSet->setAttributeAndModes(material, StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON);

    return stateSet;
}

//----------------------------------------------------------------------
StateSet *coVRCoordinateAxis::makeGreenStateSet()
{
    Material *material = new Material();

    material->setColorMode(Material::OFF);
    material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.0, 1.0, 0.0, 1.0));
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.0, 0.2, 0.0, 1.0));
    material->setAlpha(Material::FRONT_AND_BACK, 1.0);

    StateSet *stateSet = new StateSet();
    stateSet->setRenderingHint(StateSet::OPAQUE_BIN);
    stateSet->setAttributeAndModes(material, StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON);

    return stateSet;
}

//----------------------------------------------------------------------
StateSet *coVRCoordinateAxis::makeBlueStateSet()
{
    Material *material = new Material();

    material->setColorMode(Material::OFF);
    material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 1.0, 1.0));
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.2, 1.0));
    material->setAlpha(Material::FRONT_AND_BACK, 1.0);

    StateSet *stateSet = new StateSet();
    stateSet->setRenderingHint(StateSet::OPAQUE_BIN);
    stateSet->setAttributeAndModes(material, StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON);

    return stateSet;
}
