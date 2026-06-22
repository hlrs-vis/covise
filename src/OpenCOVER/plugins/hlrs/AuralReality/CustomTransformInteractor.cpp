/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CustomTransformInteractor.h"
#include <OpenVRUI/coInteraction.h>
#include <OpenVRUI/coInteractionManager.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <cover/coVRFileManager.h>
#include <cover/input/input.h>
#include <osg/Math>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <cover/coVRNavigationManager.h>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <osg/Vec3>
#include <osg/io_utils>
#include <vrb/client/SharedState.h>

const float ArrowLength = 5.0f;

using namespace opencover;

CustomTransformInteractor::CustomTransformInteractor(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium)
    : coVRIntersectionInteractor(s, type, iconName, interactorName, priority, true)
    , interactionRelative("SpaceMouse", vrui::coInteraction::InteractionType::NoButton, vrui::coInteraction::InteractionPriority::Highest)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new CustomTransformInteractor(%s)\n", interactorName);
    }

    interactionRelative.setGroup(coInteraction::GroupNavigation);

    createGeometry();

    CustomTransformInteractor::updateTransform(m);
}

CustomTransformInteractor::~CustomTransformInteractor()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~CustomTransformInteractor\n");
}

void CustomTransformInteractor::createGeometry()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "\nCustomTransformInteractor::createGeometry\n");

    axisTransform = new osg::MatrixTransform();
    axisTransform->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    scaleTransform->addChild(axisTransform);

    geometryNode = axisTransform.get();

    osg::ref_ptr<osg::Group> dummy = new osg::Group;
    auto icon = coVRFileManager::instance()->loadFile("share/covise/icons/cursors.glb", nullptr, dummy, "", true);
    icon->setNodeMask(icon->getNodeMask() | Isect::Pick | Isect::Intersection | Isect::Visible);
    axisTransform->addChild(icon);

    m_debugTarget = new osg::MatrixTransform;
    cover->getObjectsRoot()->addChild(m_debugTarget);

    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 0.1);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    auto sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereDrawable->setColor(osg::Vec4(1, 0, 1, 1));
    auto sphereGeode = new osg::Geode();
    sphereGeode->addDrawable(sphereDrawable);
    sphereGeode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    m_debugTarget->addChild(sphereGeode);
    m_debugTarget->setNodeMask(0x0);
}

void CustomTransformInteractor::updateSharedState()
{
#if 0
    if (auto st = static_cast<SharedMatrix *>(m_sharedState.get()))
    {
        *st = _oldInteractorXformMat_o; // myPosition
    }
#endif
}

void CustomTransformInteractor::startInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\nCustomTransformInteractor::startInteraction\n");

    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix o_to_w = cover->getBaseMat();
    osg::Matrix hm = getPointerMat(); // hand matrix weltcoord
    osg::Matrix hm_o = hm * w_to_o; // hand matrix objekt coord

    const std::string &name = _hitNode->getName();
    m_rotateOnly = name == "Rotate";
    m_translateOnlyAxis = name == "XArrow" ? 1 : name == "YArrow" ? 2
        : name == "ZArrow"                                        ? 3
                                                                  : 0;
    auto target = osg::Matrix::translate(_hitPos);
    m_startMatrix = getMatrix();
    m_objectToTarget = target * osg::Matrix::inverse(m_startMatrix);
    m_handToTarget = target * osg::Matrix::inverse(hm_o);

    m_objectRotation.makeIdentity();

    if (!interactionRelative.isRegistered())
        vrui::coInteractionManager::the()->registerInteraction(&interactionRelative);

    coVRIntersectionInteractor::startInteraction();
}

float applyDeadzone(float x, float d = 0.1f)
{
    float a = osg::absolute(x);
    return osg::sign(x) * (a > d ? (a - d) / (1.0 - d) : 0.0);
}

void CustomTransformInteractor::doInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\nCustomTransformInteractor::move\n");

    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix o_to_w = cover->getBaseMat();
    osg::Matrix hm = getPointerMat(); // hand matrix weltcoord
    osg::Matrix hm_o = hm * w_to_o; // hand matrix objekt coord

    auto turnValuator = Input::instance()->getValuator("RightJoyX");
    auto pushValuator = Input::instance()->getValuator("RightJoyY");
    float push = pushValuator ? applyDeadzone(pushValuator->getValue()) : 0.f;
    float turn = turnValuator ? applyDeadzone(turnValuator->getValue()) : 0.f;

    if (push != 0 && abs(push) > abs(turn))
    {
        m_handToTarget.setTrans(m_handToTarget.getTrans() * (1.0 + push * cover->frameDuration()));
    }

    if (turn != 0 && abs(turn) > abs(push))
    {
        m_objectRotation.postMult(osg::Matrix::rotate(turn * cover->frameDuration() * 5.f, osg::Vec3d(0, 0, 1)));
    }

    auto target = m_handToTarget * hm_o;
    m_debugTarget->setMatrix(target);

    osg::Matrix result;
    if (m_rotateOnly)
    {
        auto m = getMatrix();
        auto pos = m.getTrans();
        auto dir = target.getTrans() - pos;
        dir.normalize();

        auto rot = osg::Matrix::rotate(M_PI_2, osg::Vec3(1, 0, 0)) * osg::Matrix::inverse(osg::Matrix::lookAt(osg::Vec3d(0, 0, 0), dir, osg::Vec3d(0, 0, -1)));

        result = rot * osg::Matrix::translate(pos);
    }
    else if (m_translateOnlyAxis)
    {
        auto relativeMovement = osg::Matrix::inverse(m_startMatrix) * target;

        auto translation = relativeMovement.getTrans();
        translation.x() *= m_translateOnlyAxis == 1;
        translation.y() *= m_translateOnlyAxis == 2;
        translation.z() *= m_translateOnlyAxis == 3;

        result = osg::Matrix::translate(translation) * m_startMatrix;
    }
    else
    {
        result = m_objectRotation * osg::Matrix::inverse(m_objectToTarget) * target;
    }

    m_debugTarget->setNodeMask(Isect::Visible);

    // if (coVRNavigationManager::instance()->isSnapping())
    // {
    //     if (coVRNavigationManager::instance()->isDegreeSnapping())
    //     {
    //         // snap orientation
    //         snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &interactorXformMat_o);
    //     }
    //     else
    //     {
    //         // snap orientation to 45 degree
    //         snapTo45Degrees(&interactorXformMat_o);
    //     }
    // }
    //
    // and now we apply it
    updateTransform(result);
}

void CustomTransformInteractor::stopInteraction()
{
    m_debugTarget->setNodeMask(0x0);

    if (interactionRelative.isRegistered())
        vrui::coInteractionManager::the()->unregisterInteraction(&interactionRelative);

    coVRIntersectionInteractor::stopInteraction();
}

void CustomTransformInteractor::updateTransform(osg::Matrix m)
{
    if (cover->debugLevel(5))
        fprintf(stderr, "CustomTransformInteractor:setMatrix\n");

    moveTransform->setMatrix(m);

    if (m_sharedState)
    {
        if (auto st = static_cast<SharedMatrix *>(m_sharedState.get()))
        {
            *st = m;
        }
    }
}

void CustomTransformInteractor::setShared(bool shared)
{
#if 0
    if (shared)
    {
        if (!m_sharedState)
        {
            m_sharedState.reset(new SharedMatrix("interactor." + std::string(_interactorName), _oldInteractorXformMat_o)); // myPosition
            m_sharedState->setUpdateFunction([this]()
                {
                m_isInitializedThroughSharedState = true;
                osg::Matrix interactorXformMat_o = *static_cast<SharedMatrix *>(m_sharedState.get());
                if (cover->restrictOn())
                {
                    // restrict to visible scene
                    osg::Vec3 pos_o, restrictedPos_o;
                    pos_o = interactorXformMat_o.getTrans();
                    restrictedPos_o = restrictToVisibleScene(pos_o);
                    interactorXformMat_o.setTrans(restrictedPos_o);
                }

                if (coVRNavigationManager::instance()->isSnapping())
                {
                    if (coVRNavigationManager::instance()->isDegreeSnapping())
                    {
                        // snap orientation
                        snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &interactorXformMat_o);
                    }
                    else
                    {
                        // snap orientation to 45 degree
                        snapTo45Degrees(&interactorXformMat_o);
                    }
                }
                updateTransform(interactorXformMat_o); });
        }
    }
    else
    {
        m_sharedState.reset(nullptr);
    }
#endif
}
