#include "coVR3DGizmoType.h"
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/MatrixTransform>
#include <cover/coVRNavigationManager.h>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <osg/io_utils>
#include <vrbclient/SharedState.h>

#include <PluginUtil/coVR3DGizmo.h>



using namespace opencover;


coVR3DGizmoType::coVR3DGizmoType(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium, coVR3DGizmo *gizmoPointer)
    :coVRIntersectionInteractor(s, type, iconName, interactorName, priority, true),_observer{gizmoPointer}
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR3DGizmoType(%s)\n", interactorName);
    }

    _helperPlane.reset(new opencover::coPlane(osg::Vec3(0.0, 0.0, 1.0), osg::Vec3(0.0, 0.0, 0.0)));
    _helperLine.reset(new opencover::coLine(osg::Vec3(0.0, 0.0, 0.0), osg::Vec3(0.0, 0.0, 1.0)));

    _interactionB.reset(new coCombinedButtonInteraction(coInteraction::ButtonC, "ChangeGizmo",coInteraction::InteractionPriority::Medium));


    updateTransform(m);
}

coVR3DGizmoType::~coVR3DGizmoType()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~coVR3DGizmoType\n");

}

int coVR3DGizmoType::hit(vrui::vruiHit *hit)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "coVR3DGizmoType(%s)::hit\n", _interactorName);

    if(!_interactionB->isRegistered())   
       vrui::coInteractionManager::the()->registerInteraction(_interactionB.get());   // _interactionB->setHitByMouse(hit->isMouseHit());      // std::cout <<"registered"<<std::endl;
    
    return coVRIntersectionInteractor::hit(hit);


}

void coVR3DGizmoType::miss()
{
    if(_interactionB->isRegistered())
       vrui::coInteractionManager::the()->unregisterInteraction(_interactionB.get());

    coVRIntersectionInteractor::miss();
}

void coVR3DGizmoType::update()
{
    if (_interactionB->wasStopped())
        _changeGizmoType = true;

    coCombinedButtonInteraction::update();
}

void coVR3DGizmoType::preFrame()
{
    
    if(_changeGizmoType)    
        changeGizmoType();
    else
        coVRIntersectionInteractor::preFrame();

}


void coVR3DGizmoType::changeGizmoType()
{
    if(_observer != nullptr)
        _observer->changeGizmoType();

    _changeGizmoType = false;

}

void coVR3DGizmoType::startInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DGizmoType::startInteraction\n");
    
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat = getPointerMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;


    _oldInterMat_o = _interMat_o;
    _startInterMat_o = _interMat_o;

    _diff = _interMat_o.getTrans() - _hitPos;
    _distance = (_hitPos - currHandMat_o.getTrans()).length();

    coVRIntersectionInteractor::startInteraction();

}

void coVR3DGizmoType::stopInteraction()
{}
void coVR3DGizmoType::updateTransform(osg::Matrix m)
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coVR3DGizmoType:setMatrix\n");
    _interMat_o = m;
    moveTransform->setMatrix(m);
    
    //brauch man das ? 
    /*if (m_sharedState)
    {
        if (auto st = static_cast<SharedMatrix *>(m_sharedState.get()))
        {
            *st = m;
        }
    }
    */
}

void coVR3DGizmoType::updateSharedState()
{
    if (auto st = static_cast<SharedMatrix *>(m_sharedState.get()))
    {
        *st = _oldInterMat_o;
    }
}

void coVR3DGizmoType::setShared(bool shared)
{

}

// Der funktionsname passt nicht zu dem was es tut ? Bekommen nicht den Schnittpunkt sondern startPos! 
osg::Vec3 coVR3DGizmoType::calcPlaneLineIntersection(const osg::Vec3& lp0, const osg::Vec3& lp1, osg::Vec3 fixAxis_o) const
{
    osg::Vec3 isectPoint, newPos;
    _helperPlane->update(fixAxis_o , getMatrix().getTrans()); // FIXME: fixAxis * getMatrix !?
    bool intersect = _helperPlane->getLineIntersectionPoint( lp0, lp1, isectPoint);
    newPos  = isectPoint + _diff;

    // hier aufpassen, dass man nicht axe schon multipliziert reingibt !
    if(fixAxis_o == osg::X_AXIS)
        newPos.x() = _oldInterMat_o.getTrans().x();
    else if(fixAxis_o == osg::Y_AXIS)
        newPos.y() = _oldInterMat_o.getTrans().y();
    else if(fixAxis_o == osg::Z_AXIS)
        newPos.z() = _oldInterMat_o.getTrans().z();

    return newPos; //FIXME what happens if lines are parallel ? 
}

void coVR3DGizmoType::calculatePointerDirection_w(osg::Vec3& lp0, osg::Vec3& lp1, osg::Vec3& pointerDir ) const
{
    osg::Vec3 origin(0, 0, 0);
    osg::Vec3 yaxis(0, 1, 0);
    osg::Matrix o_to_w = cover->getBaseMat();
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat = getPointerMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;

    // pointer direction in world coordinates
    lp0 = origin * currHandMat;
    lp1 = yaxis *currHandMat;
    pointerDir = lp1 - lp0;
    pointerDir.normalize();
}   


void coVR3DGizmoType::calculatePointerDirection_o(osg::Vec3& lp0_o, osg::Vec3& lp1_o, osg::Vec3& pointerDir_o ) const
{
    osg::Vec3 origin(0, 0, 0);
    osg::Vec3 yaxis(0, 1, 0);
    osg::Matrix o_to_w = cover->getBaseMat();
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat = getPointerMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;

    // pointer direction in world coordinates
    osg::Vec3 lp0 = origin * currHandMat;
    osg::Vec3 lp1 = yaxis *currHandMat;

    // pointer direction in object coordinates
    lp0_o = lp0 * w_to_o;
    lp1_o = lp1 * w_to_o;
    pointerDir_o = lp1_o - lp0_o;
    pointerDir_o.normalize();   
}
