/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVR3DTransInteractor.h"
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRNavigationManager.h>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>

using namespace opencover;

coVR3DTransInteractor::coVR3DTransInteractor(osg::Vec3 p, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium)
    : coVRIntersectionInteractor(s, type, iconName, interactorName, priority)
{
    osg::Matrix m;

    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR3DTransInteractor(%s)\n", interactorName);
    }

    _interPos = p;

    m.makeTranslate(_interPos);
    moveTransform->setMatrix(m);
    geometryNode = new osg::Geode();
    scaleTransform->addChild(geometryNode.get());

    createGeometry();
}

coVR3DTransInteractor::~coVR3DTransInteractor()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~coVR3DTransInteractor");
}

void
coVR3DTransInteractor::createGeometry()
{
    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.0);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    mySphereDrawable->setColor(osg::Vec4(0.6f, 0.6f, 0.6f, 1.0f));
    ((osg::Geode *)geometryNode.get())->addDrawable(mySphereDrawable);
    geometryNode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
}

void
coVR3DTransInteractor::startInteraction()
{
    osg::Matrix currentHandMat;
    osg::Vec3 currentHandPos, currentHandPos_o;

    if (cover->debugLevel(2))
        fprintf(stderr, "\ncoVR3DTransInteractor::startInteraction\n");

    // get diff between intersection point and sphere center
    _diff = _interPos - _hitPos;

    currentHandMat = getPointerMat();
    _oldHandMat = currentHandMat;
    currentHandPos = currentHandMat.getTrans();
    currentHandPos_o = currentHandPos * cover->getInvBaseMat();
    _d = (_hitPos - currentHandPos_o).length();

    coVRIntersectionInteractor::startInteraction();
}

void
coVR3DTransInteractor::doInteraction()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ncoVR3DTransInteractor::doInteraction\n");

    osg::Matrix currentHandMat, currentHandMat_o, m;
    osg::Vec3 currentHandPos, currentHandPos_o, pointerDir_o, origin(0, 0, 0), yaxis(0, 1, 0);
    osg::Vec3 lp1, lp2, lp1_o, lp2_o;

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
        _oldHandMat = currentHandMat;
    }
    else
    {
        lp1 = origin * currentHandMat;
        lp2 = yaxis * currentHandMat;
        lp1_o = lp1 * cover->getInvBaseMat();
        lp2_o = lp2 * cover->getInvBaseMat();

        pointerDir_o = lp2_o - lp1_o;
        pointerDir_o.normalize();

        // get hand pos in object coords
        currentHandPos = currentHandMat.getTrans();
        currentHandPos_o = currentHandPos * cover->getInvBaseMat();

        _interPos = currentHandPos_o + pointerDir_o * _d + _diff;
    }

    if (cover->restrictOn())
    {
        // restrict to visible scene
        _interPos = restrictToVisibleScene(_interPos);
    }
    if (_interPos == osg::Vec3(0.0, 0.0, 0.0))
        _interPos.set(0.0, 0.0, 0.00001);
    m.makeTranslate(_interPos);
    moveTransform->setMatrix(m);
}

void
coVR3DTransInteractor::updateTransform(osg::Vec3 pos)
{
    if (pos == osg::Vec3(0.0, 0.0, 0.0))
        pos.set(0.0, 0.0, 0.00001);

    _interPos = pos;
    osg::Matrix m;

    m.makeIdentity();
    m.setTrans(pos);

    moveTransform->setMatrix(m);
}
