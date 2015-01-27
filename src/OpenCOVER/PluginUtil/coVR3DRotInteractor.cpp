/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVR3DRotInteractor.h"
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRPluginSupport.h>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>

namespace opencover
{

coVR3DRotInteractor::coVR3DRotInteractor(osg::Vec3 rotationPoint, osg::Vec3 position, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority)
    : coVRIntersectionInteractor(s, type, iconName, interactorName, priority)
{
    osg::Matrix m;

    if (cover->debugLevel(2))
        fprintf(stderr, "new coVR3DRotInteractor(%s)\n", interactorName);

    moveTransform->setMatrix(m);
    geometryNode = new osg::Geode();
    scaleTransform->addChild(geometryNode.get());

    createGeometry();

    updateTransform(rotationPoint, position);
}

coVR3DRotInteractor::~coVR3DRotInteractor()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~coVR3DRotInteractor");
}

void coVR3DRotInteractor::createGeometry()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "coVR3DRotInteractor createGeometry\n");

    osg::Cone *myCone = new osg::Cone(osg::Vec3(0, 0, 0), 1.2, 2.5);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *myConeDrawable = new osg::ShapeDrawable(myCone, hint);
    myConeDrawable->setColor(osg::Vec4(0.6f, 0.6f, 0.6f, 1.0f));
    ((osg::Geode *)geometryNode.get())->addDrawable(myConeDrawable);
    geometryNode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
}

void coVR3DRotInteractor::startInteraction()
{
    osg::Matrix currentHandMat;
    osg::Vec3 currentHandPos, currentHandPos_o;

    if (cover->debugLevel(2))
        fprintf(stderr, "\ncoVR3DRotInteractor::startInteraction\n");

    // get diff between intersection point and position
    _diff = _interPos - _hitPos;

    currentHandMat = getPointerMat();
    _oldHandMat = currentHandMat;
    currentHandPos = currentHandMat.getTrans();
    currentHandPos_o = currentHandPos * cover->getInvBaseMat();
    _d = (_hitPos - currentHandPos_o).length();

    coVRIntersectionInteractor::startInteraction();
}

void coVR3DRotInteractor::doInteraction()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ncoVR3DRotInteractor::doInteraction\n");
    osg::Matrix currentHandMat, currentHandMat_o, m;
    osg::Vec3 currentHandPos, currentHandPos_o, pointerDir_o, origin(0, 0, 0), yaxis(0, 1, 0);
    osg::Vec3 lp1, lp2, lp1_o, lp2_o;

    // get hand pos in object coords
    currentHandMat = getPointerMat();

    if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRConfig::instance()->useWiiNavigationVisenso())
    {
        osg::Vec3 trans = currentHandMat.getTrans();
        trans[1] = _oldHandMat.getTrans()[1];
        currentHandMat.setTrans(trans);

        osg::Matrix _oldHandMat_o = _oldHandMat * cover->getInvBaseMat();
        currentHandMat_o = currentHandMat * cover->getInvBaseMat();

        trans = (currentHandMat_o.getTrans() - _oldHandMat_o.getTrans());
        currentHandPos = currentHandMat.getTrans();
        currentHandPos_o = currentHandPos * cover->getInvBaseMat();
        _interPos = _interPos + trans;
    }
    else
    {

        lp1 = origin * currentHandMat;
        lp2 = yaxis * currentHandMat;
        lp1_o = lp1 * cover->getInvBaseMat();
        lp2_o = lp2 * cover->getInvBaseMat();

        pointerDir_o = lp2_o - lp1_o;
        pointerDir_o.normalize();

        currentHandPos = currentHandMat.getTrans();
        currentHandPos_o = currentHandPos * cover->getInvBaseMat();

        _interPos = currentHandPos_o + pointerDir_o * _d + _diff;
    }

    _oldHandMat = currentHandMat;

    if (_interPos == osg::Vec3(0.0, 0.0, 0.0))
        _interPos.set(0.0, 0.0, 0.00001);

    m.makeTranslate(_interPos);
    moveTransform->setMatrix(m);
}

void coVR3DRotInteractor::updateTransform(osg::Vec3 rotationPoint, osg::Vec3 pos)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "updateTransform %s \n", _interactorName);

    if (pos == osg::Vec3(0.0, 0.0, 0.0))
        pos.set(0.0, 0.0, 0.00001);

    _interPos = pos;
    _rotationPoint = rotationPoint;

    osg::Matrix m;

    // rotate from cone position to new position, rotate around rotation point
    m.makeRotate(osg::Vec3(0.0, 0.0, 1.0) - osg::Vec3(0.0, 0.0, 0.0), _interPos - _rotationPoint);
    m.setTrans(pos);
    //for(int i=0;i<4;i++)
    //fprintf(stderr,"coVR3DRotInteractor::updateTransform Matrix %f %f %f %f\n", m(i,0),m(i,1),m(i,2),m(i,3));
    moveTransform->setMatrix(m);
}

void coVR3DRotInteractor::updateRotationPoint(osg::Vec3 rotPoint)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "coVR3DRotInteractor::updateRotationPoint %s \n", _interactorName);

    _rotationPoint = rotPoint;
    updateTransform(_rotationPoint, _interPos);
}
}
