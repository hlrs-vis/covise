/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVR2DTransInteractor.h"
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <osg/ShapeDrawable>

namespace opencover
{

coVR2DTransInteractor::coVR2DTransInteractor(osg::Vec3 point, osg::Vec3 normal, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, enum coInteraction::InteractionPriority priority = Medium)
    : coVRIntersectionInteractor(s, type, iconName, interactorName, priority)
    , coPlane(normal, point)
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "\nnew coVR2DTransInteractor(%s) of size %f\n", interactorName, s);
    }

    osg::Matrix m;

    // Don't rotate the interactor anymore. Since we usually have a sphere, it doesn't matter.
    // Interactors inheriting from this class might have a problem with the rotation (e.g. ChemicalReactionPlugin).
    //osg::Vec3 zaxis(0,0,1);
    //m.makeRotate(zaxis, _normal);

    m.makeTranslate(point);
    moveTransform->setMatrix(m);

    createGeometry();
}

coVR2DTransInteractor::~coVR2DTransInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\ndelete coVR2DTransInteractor\n");
}

void
coVR2DTransInteractor::createGeometry()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVR2DTransInteractor::createGeometry\n");

    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.0);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    mySphereDrawable->setColor(osg::Vec4(0.6f, 0.6f, 0.6f, 1.0f));
    geometryNode = new osg::Geode();
    scaleTransform->addChild(geometryNode.get());
    ((osg::Geode *)geometryNode.get())->addDrawable(mySphereDrawable);
    geometryNode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
}

void
coVR2DTransInteractor::startInteraction()
{

    // store vector between hit point and cylinder center in object coordinates
    osg::Vec3 lp0(0.0, 0.0, 0.0), lp1(0.0, 1.0, 0.0), isectPoint;

    // pointer direction in world coordinates
    lp0 = lp0 * getPointerMat();
    lp1 = lp1 * getPointerMat();

    // pointer direction in object coordinates
    lp0 = lp0 * cover->getInvBaseMat();
    lp1 = lp1 * cover->getInvBaseMat();

    // get intersection point in object coordinates
    if (getLineIntersectionPoint(lp0, lp1, isectPoint))
    {
        _diff = _point - isectPoint;
    }

    coVRIntersectionInteractor::startInteraction();
}

void
coVR2DTransInteractor::doInteraction()
{
    osg::Matrix m;
    osg::Vec3 lp0(0.0, 0.0, 0.0), lp1(0.0, 1.0, 0.0), isectPoint;

    // pointer direction in world coordinates
    lp0 = lp0 * getPointerMat();
    lp1 = lp1 * getPointerMat();

    // pointer direction in object coordinates
    lp0 = lp0 * cover->getInvBaseMat();
    lp1 = lp1 * cover->getInvBaseMat();

    // get intersection point in object coordinates
    if (getLineIntersectionPoint(lp0, lp1, isectPoint))
    {
        _point = isectPoint + _diff;
        if (cover->restrictOn())
        {
            // restrict to visible scene
            _point = restrictToVisibleScene(_point);
        }
    }

    //osg::Vec3 zaxis(0,0,1);
    //m.makeRotate(zaxis, _normal);
    if (_point == osg::Vec3(0.0, 0.0, 0.0))
        _point.set(0.0, 0.0, 0.00001);
    m.makeTranslate(_point);
    moveTransform->setMatrix(m);
}

void
coVR2DTransInteractor::updateTransform(osg::Vec3 pos, osg::Vec3 normal)
{
    if (pos == osg::Vec3(0.0, 0.0, 0.0))
        pos.set(0.0, 0.0, 0.00001);
    coPlane::update(normal, pos);
    osg::Matrix m;
    //osg::Vec3 zaxis(0, 0, 1);
    //m.makeRotate(zaxis, _normal);
    m.makeTranslate(_point);
    moveTransform->setMatrix(m);
}
}
