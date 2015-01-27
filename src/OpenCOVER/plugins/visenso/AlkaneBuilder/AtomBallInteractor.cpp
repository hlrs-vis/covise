/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AtomBallInteractor.h"
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/coPlane.h>
#include <osg/ShapeDrawable>

AtomBallInteractor::AtomBallInteractor(string symbol, const char *interactorName, osg::Vec3 initialPos, osg::Vec3 normal, float size, osg::Vec4 color)
    : coVR2DTransInteractor(initialPos, normal, size, coInteraction::ButtonA, "Hand", interactorName, coInteraction::Medium)
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "AtomBallInteractor::AtomBallInteractor\n");

    initialPos_ = lastPos_ = initialPos;
    symbol_ = symbol;
    color_ = color;
    oldInvMat_.invert(getMatrix());

    createGeometry();
}

AtomBallInteractor::~AtomBallInteractor()
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "AtomBallInteractor::~AtomBallInteractor\n");
}

void AtomBallInteractor::createGeometry()
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "AtomBallInteractor::createGeometry\n");

    //sphere with radius 0.5*size
    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 0.5 * _interSize);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereDrawable->setColor(osg::Vec4(color_[0], color_[1], color_[2], color_[3]));
    osg::Geode *sphereGeode = new osg::Geode();
    sphereGeode->addDrawable(sphereDrawable);
    osg::StateSet *normalState = opencover::VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE);
    sphereGeode->setStateSet(normalState);
    group_ = new osg::Group();
    group_->addChild(sphereGeode);

    //coVRIntersectionInteractor::geometryNode
    geometryNode = group_;

    // remove old geometry
    scaleTransform->removeChild(0, scaleTransform->getNumChildren());

    // add geometry to node and remove old scaling
    scaleTransform->addChild(group_);
    osg::Matrix s;
    scaleTransform->setMatrix(s);
}

void AtomBallInteractor::preFrame()
{
    //fprintf(stderr,"AtomBallInteractor::preFrame----\n");

    coVRIntersectionInteractor::preFrame();

    // preFrame does a keepSize,
    // but this interactor should be smaller if far away, therefore undo scaling
    osg::Matrix s;
    scaleTransform->setMatrix(s);
}
void AtomBallInteractor::startInteraction()
{
    lastPos_ = getPosition();
    coVR2DTransInteractor::doInteraction();
    osg::Vec3 restrictedPos = restrictToBox(getPosition());
    if (restrictedPos != getPosition())
    {
        updateTransform(restrictedPos, getNormal());
        lastPos_ = restrictedPos; //vermeide abbreisen durch restriction
    }
}
void AtomBallInteractor::doInteraction()
{
    lastPos_ = getPosition();
    coVR2DTransInteractor::doInteraction();
    osg::Vec3 restrictedPos = restrictToBox(getPosition());
    if (restrictedPos != getPosition())
    {
        updateTransform(restrictedPos, getNormal());
        lastPos_ = restrictedPos; //vermeide abbreisen durch restriction
    }
}
void AtomBallInteractor::stopInteraction()
{
    lastPos_ = getPosition();
    coVR2DTransInteractor::doInteraction();
    osg::Vec3 restrictedPos = restrictToBox(getPosition());
    if (restrictedPos != getPosition())
    {
        updateTransform(restrictedPos, getNormal());
        lastPos_ = restrictedPos; //vermeide abbreisen durch restriction
    }
}
void
AtomBallInteractor::resetPosition()
{
    // coPlane::_normal
    updateTransform(initialPos_, getNormal());
}

osg::Matrix
AtomBallInteractor::getDiffMat()
{

    return diffMat_;
}

void
AtomBallInteractor::updateTransform(osg::Matrix mat)
{

    coVR2DTransInteractor::updateTransform(mat.getTrans(), getNormal());
    diffMat_ = oldInvMat_ * getMatrix();
    oldInvMat_.invert(getMatrix());
}

void
AtomBallInteractor::updateTransform(osg::Vec3 p, osg::Vec3 n)
{

    coVR2DTransInteractor::updateTransform(p, n);
    diffMat_ = oldInvMat_ * getMatrix();
    oldInvMat_.invert(getMatrix());
}

osg::Vec3
AtomBallInteractor::restrictToBox(osg::Vec3 pos)
{
    //fprintf(stderr,"AtomBallInteractor::restrictToBox box=[%f %f %f][%f %f %f]\n",box_.xMin(), box_.yMin(), box_.zMin(), box_.xMax(), box_.yMax(), box_.zMax());

    if (!box_.valid())
    {
        return (pos);
    }
    osg::Vec3 rpos = pos;
    if (pos[0] < box_.xMin())
    {
        rpos[0] = box_.xMin();
        //fprintf(stderr, "restricting posx=[%f] to box minx=[%f]\n", pos[0], box.min[0]);
    }
    if (pos[1] < box_.yMin())
    {
        rpos[1] = box_.yMin();
        //fprintf(stderr, "restricting posy=[%f] to box miny=[%f]\n", pos[1], box.min[1]);
    }
    if (pos[2] < box_.zMin())
    {
        rpos[2] = box_.zMin();
        //fprintf(stderr, "restricting posz=[%f] to box minz=[%f]\n", pos[2], box.min[2]);
    }
    if (pos[0] > box_.xMax())
    {
        rpos[0] = box_.xMax();
        //fprintf(stderr, "restricting posx=[%f] to box maxx=[%f]\n", pos[0], box.max[0]);
    }
    if (pos[1] > box_.yMax())
    {
        rpos[1] = box_.yMax();
        //fprintf(stderr, "restricting posy=[%f] to box maxy=[%f]\n", pos[1], box.max[1]);
    }
    if (pos[2] > box_.zMax())
    {
        rpos[2] = box_.zMax();
        //fprintf(stderr, "restricting posz=[%f] to box maxz=[%f]\n", pos[2], box.max[2]);
    }
    //fprintf(stderr,"AtomBallInteractor::restrictToBox pos=[%f %f %f] rpos=[%f %f %f]\n", pos[0], pos[1], pos[2], rpos[0], rpos[1], rpos[2]);
    return rpos;
}
