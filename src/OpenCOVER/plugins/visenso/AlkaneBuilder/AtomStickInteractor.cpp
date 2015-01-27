/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AtomStickInteractor.h"
#include <osg/ShapeDrawable>
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>

AtomStickInteractor::AtomStickInteractor(string symbol, const char *interactorName, Atom *myAtom, osg::Matrix initialOrientation, osg::Vec3 initialPos, osg::Vec3 stickDir, float size, osg::Vec4 color)
    : coVR3DRotCenterInteractor(initialOrientation, initialPos, size, coInteraction::ButtonA, "Hand", interactorName, coInteraction::Medium)
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "AtomStickInteractor::AtomStickInteractor\n");

    initialOrientation_ = initialOrientation;
    initialPos_ = initialPos;
    dir_ = stickDir;
    dir_.normalize();
    symbol_ = symbol;
    color_ = color;
    myAtom_ = myAtom;

    osg::Vec3 zaxis(0, 0, 1);
    osg::Matrix mr;
    mr.makeRotate(zaxis, dir_);
    mr.postMultTranslate(dir_ * 0.5 * _interSize);
    // cylinder of lengths size, beginning at center of sphere,
    //sphere is 0,5*size so also 0.5*size of the cylinder comes out of the spehere
    osg::Cylinder *myCylinder = new osg::Cylinder(osg::Vec3(0, 0, 0), 0.2 * _interSize, _interSize);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *cylinderDrawable = new osg::ShapeDrawable(myCylinder, hint);
    cylinderDrawable->setColor(osg::Vec4(color[0], color[1], color[2], color[3]));
    osg::Geode *cylinderGeode = new osg::Geode();
    osg::StateSet *normalState = opencover::VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE);
    cylinderGeode->setStateSet(normalState);

    transform_ = new osg::MatrixTransform();
    transform_->setMatrix(mr);
    cylinderGeode->addDrawable(cylinderDrawable);
    transform_->addChild(cylinderGeode);

    //coVRIntersectionInteractor::geometryNode
    geometryNode = transform_;

    // remove old geometry
    scaleTransform->removeChild(0, scaleTransform->getNumChildren());

    // add geometry to node and remove old scaling
    scaleTransform->addChild(transform_);
    osg::Matrix s;
    scaleTransform->setMatrix(s);

    connectedStick_ = NULL; //not connected

    oldInvMat_.invert(getMatrix());
}

AtomStickInteractor::~AtomStickInteractor()
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "AtomStickInteractor::~AtomStickInteractor\n");

    connectedStick_ = NULL; //not connected
}

void AtomStickInteractor::preFrame()
{
    //fprintf(stderr,"AtomInteractor::preFrame----\n");
    coVR3DRotCenterInteractor::preFrame();

    // preFrame does a keepSize,
    // but this interactor should be smaller if far away, therefore undo scaling
    osg::Matrix s;
    scaleTransform->setMatrix(s);

    diffMat_ = oldInvMat_ * getMatrix();
    oldInvMat_.invert(getMatrix());
}

osg::Vec3
AtomStickInteractor::getDir()
{
    return dir_;
}

void
AtomStickInteractor::resetPosition()
{
    updateTransform(initialOrientation_, initialPos_);
}

void
AtomStickInteractor::setConnectedStick(AtomStickInteractor *s)
{
    //fprintf(stderr,"AtomConnection::setConnectedConnection\n");
    connectedStick_ = s;
}

AtomStickInteractor *
AtomStickInteractor::getConnectedStick()
{
    return connectedStick_;
}

Atom *
AtomStickInteractor::getAtom()
{
    return myAtom_;
}

osg::Matrix
AtomStickInteractor::getDiffMat()
{
    return diffMat_;
}

void AtomStickInteractor::updateTransform(osg::Matrix m, osg::Vec3 p)
{
    coVR3DRotCenterInteractor::updateTransform(m, p);

    diffMat_ = oldInvMat_ * getMatrix();
    oldInvMat_.invert(getMatrix());
}
