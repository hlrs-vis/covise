/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVR1DTransInteractor.h"
#include <cover/VRSceneGraph.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <osg/ShapeDrawable>

namespace opencover
{

coVR1DTransInteractor::coVR1DTransInteractor(osg::Vec3 point, osg::Vec3 normal, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, enum coInteraction::InteractionPriority priority = Medium)
    : coVRIntersectionInteractor(s, type, iconName, interactorName, priority)
{
    _normal = normal;
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR1DTransInteractor(%s)\n", interactorName);
    }

    osg::Matrix m;
    m.setTrans(point);
    moveTransform->setMatrix(m);

    createGeometry();
}

coVR1DTransInteractor::~coVR1DTransInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\ndelete coVR1DTransInteractor\n");
}

void
coVR1DTransInteractor::createGeometry()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "\ncoVR1DTransInteractor::createSphere\n");

    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.0);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    mySphereDrawable->setColor(osg::Vec4(0.6f, 0.6f, 0.6f, 1.0f));
    geometryNode = new osg::Geode();
    scaleTransform->addChild(geometryNode.get());
    ((osg::Geode *)geometryNode.get())->addDrawable(mySphereDrawable);
    geometryNode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
    geometryNode->setNodeMask(geometryNode->getNodeMask() | (Isect::Pick) | (Isect::Intersection));
}

void
coVR1DTransInteractor::startInteraction()
{

    // get diff between intersection point and sphere center
    _diff = moveTransform->getMatrix().getTrans() - _hitPos;
    _oldHandMat = cover->getPointerMat();

    coVRIntersectionInteractor::startInteraction();
}

void
coVR1DTransInteractor::doInteraction()
{
    osg::Matrix m;
    osg::Matrix currentHandMat = cover->getPointerMat();
    if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRConfig::instance()->useWiiNavigationVisenso())
    {
        osg::Vec3 trans = currentHandMat.getTrans();
        trans[1] = _oldHandMat.getTrans()[1];
        currentHandMat.setTrans(trans);
    }
    // current pos of interactor
    osg::Vec3 currentPos = moveTransform->getMatrix().getTrans();
    // isect Vec is vec from interacotr pos to hit pos
    osg::Vec3 isectVec = _hitPos - currentPos;
    // get dot-product for projection of isectVec on to normal
    float product = _normal * isectVec;
    osg::Vec3 point = currentPos + (_normal * product);

    if (point == osg::Vec3(0.0, 0.0, 0.0))
        point.set(0.0, 0.0, 0.001);
    m.setTrans(point);
    moveTransform->setMatrix(m);
}

void
coVR1DTransInteractor::updateTransform(osg::Vec3 pos)
{
    if (pos == osg::Vec3(0.0, 0.0, 0.0))
        pos.set(0.0, 0.0, 0.00001);
    osg::Matrix m;
    m.makeTranslate(pos);
    moveTransform->setMatrix(m);
}
}
