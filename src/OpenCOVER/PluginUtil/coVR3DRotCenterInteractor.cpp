/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVR3DRotCenterInteractor.h"
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>

namespace opencover
{

coVR3DRotCenterInteractor::coVR3DRotCenterInteractor(osg::Matrix m, osg::Vec3 p, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority)
    : coVRIntersectionInteractor(s, type, iconName, interactorName, priority)
{

    if (opencover::cover->debugLevel(2))
        fprintf(stderr, "new coVR3DRotCenterInteractor(%s)\n", interactorName);

    m_ = m;
    p_ = p;
    m_(3, 0) = p_[0];
    m_(3, 1) = p_[1];
    m_(3, 2) = p_[2];

    moveTransform->setMatrix(m_);
    frameDiffMat_.makeIdentity();

    geometryNode = new osg::Geode();
    scaleTransform->addChild(geometryNode.get());

    createGeometry();
}

coVR3DRotCenterInteractor::~coVR3DRotCenterInteractor()
{
    if (opencover::cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~coVR3DRotCenterInteractor");
}

void coVR3DRotCenterInteractor::createGeometry()
{
    if (opencover::cover->debugLevel(2))
        fprintf(stderr, "\ncoVR3DRotCenterInteractor createGeometry\n");

    osg::Cone *myCone = new osg::Cone(osg::Vec3(0, 0, 0), 1.2, 2.5);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *myConeDrawable = new osg::ShapeDrawable(myCone, hint);
    myConeDrawable->setColor(osg::Vec4(0.6f, 0.6f, 0.6f, 1.0f));
    ((osg::Geode *)geometryNode.get())->addDrawable(myConeDrawable);
    geometryNode->setStateSet(opencover::VRSceneGraph::instance()->loadDefaultGeostate());
}

void coVR3DRotCenterInteractor::startInteraction()
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "\ncoVR3DRotCenterInteractor::startInteraction\n");

    osg::Matrix currentHandMat = opencover::cover->getPointerMat();
    invOldHandMat_.invert(currentHandMat);

    coVRIntersectionInteractor::startInteraction();

    startMat_ = moveTransform->getMatrix();

    frameDiffMat_.makeIdentity();
}

void coVR3DRotCenterInteractor::doInteraction()
{
    if (opencover::cover->debugLevel(2))
        fprintf(stderr, "\ncoVR3DRotCenterInteractor::doInteraction\n");

    coVRIntersectionInteractor::doInteraction();
    osg::Matrix m, h;

    osg::Matrix currentHandMat = opencover::cover->getPointerMat();
    osg::Matrix diffMat = invOldHandMat_ * currentHandMat;
    m_ = startMat_ * diffMat;

    m_(3, 0) = p_[0];
    m_(3, 1) = p_[1];
    m_(3, 2) = p_[2];

    osg::Matrix oldm, oldminv;
    oldm = moveTransform->getMatrix();
    oldminv.invert(oldm);
    moveTransform->setMatrix(m_);
    frameDiffMat_ = oldminv * m_;
}
void coVR3DRotCenterInteractor::stopInteraction()
{
    coVRIntersectionInteractor::stopInteraction();
    frameDiffMat_.makeIdentity();
}

void coVR3DRotCenterInteractor::updateTransform(osg::Matrix m, osg::Vec3 p)
{
    if (opencover::cover->debugLevel(2))
        fprintf(stderr, "updateTransform %s \n", _interactorName);

    m_ = m;
    p_ = p;
    m_(3, 0) = p_[0];
    m_(3, 1) = p_[1];
    m_(3, 2) = p_[2];

    moveTransform->setMatrix(m_);
}

void coVR3DRotCenterInteractor::updatePosition(osg::Vec3 pos)
{
    if (opencover::cover->debugLevel(2))
        fprintf(stderr, "coVR3DRotCenterInteractor::updateRotationPoint %s \n", _interactorName);

    p_ = pos;
    m_ = opencover::cover->getPointerMat();
    m_(3, 0) = p_[0];
    m_(3, 1) = p_[1];
    m_(3, 2) = p_[2];

    moveTransform->setMatrix(m_);
}

}
