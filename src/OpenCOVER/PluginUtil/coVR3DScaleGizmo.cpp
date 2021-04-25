#include "coVR3DScaleGizmo.h"
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/MatrixTransform>
#include <cover/coVRNavigationManager.h>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <osg/io_utils>
#include <vrb/client/SharedState.h>

const float ArrowLength = 5.0f;

using namespace opencover;

coVR3DScaleGizmo::coVR3DScaleGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium,coVR3DGizmo* gizmoPointer)
    :coVR3DGizmoType(m, s, type, iconName, interactorName, priority, gizmoPointer)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR3DScaleGizmo(%s)\n", interactorName);
    }

    createGeometry();
    _line.reset(new opencover::coLine(osg::Vec3(0.0, 0.0, 0.0), osg::Vec3(0.0, 0.0, 1.0)));

    coVR3DScaleGizmo::updateTransform(m);
}

coVR3DScaleGizmo::~coVR3DScaleGizmo()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~coVR3DScaleGizmo\n");
}

void
coVR3DScaleGizmo::createGeometry()
{
   if (cover->debugLevel(4))
        fprintf(stderr, "\ncoVR3DScaleGizmo::createGeometry\n");

    bool restrictedInteraction = true; 

    osg::ShapeDrawable *sphereDrawable, *xCylDrawable, *yCylDrawable, *zCylDrawable, *xSphereDrawable, *ySphereDrawable, *zSphereDrawable;

    osg::Vec3 origin(0, 0, 0), px(1, 0, 0), py(0, 1, 0), pz(0, 0, 1);
    osg::Vec3 yaxis(0, 1, 0);
    osg::Vec4 red(0.5, 0.2, 0.2, 1.0), green(0.2, 0.5, 0.2, 1.0), blue(0.2, 0.2, 0.5, 1.0), color(0.5, 0.5, 0.5, 1);
    
    axisTransform = new osg::MatrixTransform();
    axisTransform->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    geometryNode = axisTransform.get();
    scaleTransform->addChild(geometryNode.get());

    osg::Sphere *mySphere = new osg::Sphere(origin, 0.5);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereDrawable->setColor(color);
    sphereGeode = new osg::Geode();
    sphereGeode->addDrawable(sphereDrawable);
    axisTransform->addChild(sphereGeode.get());

    //create axis
    auto cyl = new osg::Cylinder(origin, 0.15, ArrowLength);
    xCylDrawable = new osg::ShapeDrawable(cyl, hint);
    yCylDrawable = new osg::ShapeDrawable(cyl, hint);
    zCylDrawable = new osg::ShapeDrawable(cyl, hint);

    xCylDrawable->setColor(red);
    yCylDrawable->setColor(green);
    zCylDrawable->setColor(blue);

    xAxisTransform = new osg::MatrixTransform();
    xAxisTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(90.), 0, 1, 0)*osg::Matrix::translate(osg::Vec3(ArrowLength, 0, 0)*0.5));
    yAxisTransform = new osg::MatrixTransform();
    yAxisTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(-90.), 1, 0, 0)*osg::Matrix::translate(osg::Vec3(0, ArrowLength, 0)*0.5));
    zAxisTransform = new osg::MatrixTransform();
    zAxisTransform->setMatrix(osg::Matrix::translate(osg::Vec3(0, 0, ArrowLength)*0.5));

    scaleXaxisGeode = new osg::Geode;
    scaleXaxisGeode->addDrawable(xCylDrawable);
    xAxisTransform->addChild(scaleXaxisGeode);
    scaleYaxisGeode = new osg::Geode;
    scaleYaxisGeode->addDrawable(yCylDrawable);
    yAxisTransform->addChild(scaleYaxisGeode);
    scaleZaxisGeode = new osg::Geode;
    scaleZaxisGeode->addDrawable(zCylDrawable);
    zAxisTransform->addChild(scaleZaxisGeode);

    axisTransform->addChild(xAxisTransform);
    axisTransform->addChild(yAxisTransform);
    axisTransform->addChild(zAxisTransform);
    

    osg::Sphere *smallSphere = new osg::Sphere(origin, 0.5);
    xSphereDrawable = new osg::ShapeDrawable(smallSphere, hint);
    ySphereDrawable = new osg::ShapeDrawable(smallSphere, hint);
    zSphereDrawable = new osg::ShapeDrawable(smallSphere, hint);

    xSphereDrawable->setColor(red);
    ySphereDrawable->setColor(green);
    zSphereDrawable->setColor(blue);

    xSphereTransform = new osg::MatrixTransform();
    xSphereTransform->setMatrix(osg::Matrix::translate(osg::Vec3(ArrowLength, 0, 0)));
    ySphereTransform = new osg::MatrixTransform();
    ySphereTransform->setMatrix(osg::Matrix::translate(osg::Vec3(0, ArrowLength, 0)));
    zSphereTransform = new osg::MatrixTransform();
    zSphereTransform->setMatrix(osg::Matrix::translate(osg::Vec3(0, 0, ArrowLength)));

    scaleXSphereGeode = new osg::Geode;
    scaleXSphereGeode->addDrawable(xSphereDrawable);
    xSphereTransform->addChild(scaleXSphereGeode);
    scaleYSphereGeode = new osg::Geode;
    scaleYSphereGeode->addDrawable(ySphereDrawable);
    ySphereTransform->addChild(scaleYSphereGeode);
    scaleZSphereGeode = new osg::Geode;
    scaleZSphereGeode->addDrawable(zSphereDrawable);
    zSphereTransform->addChild(scaleZSphereGeode);

    axisTransform->addChild(xSphereTransform);
    axisTransform->addChild(ySphereTransform);
    axisTransform->addChild(zSphereTransform);

    //create temporary axis which show scale when interactor is active
    tempxAxisTransform = new osg::MatrixTransform();
    tempyAxisTransform = new osg::MatrixTransform();
    tempzAxisTransform = new osg::MatrixTransform();
    
    axisTransform->addChild(tempxAxisTransform);
    axisTransform->addChild(tempyAxisTransform);
    axisTransform->addChild(tempzAxisTransform);
}

//wozu brauche ich das ?
/*void coVR3DScaleGizmo::updateSharedState()
{
    if (auto st = static_cast<SharedMatrix *>(m_sharedState.get()))
    {
        *st = _oldInteractorXformMat_o;
    }
}
*/
void coVR3DScaleGizmo::startInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DScaleGizmo::startInteraction\n");
    
    coVR3DGizmoType::startInteraction();

//########################################
    osg::Vec3 lp0_o, lp1_o, pointerDir_o;
    calculatePointerDirection_o(lp0_o, lp1_o, pointerDir_o);
    
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat = getPointerMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;
    osg::Vec3 currHandPos_o = currHandMat_o.getTrans();  
    osg::Vec3 interPos = currHandPos_o + pointerDir_o* _distance + _diff;
    
    osg::Matrix interactor_to_w = getMatrix();
    osg::Matrix w_to_interactor;
    w_to_interactor.invert(interactor_to_w);
    //_startHitPos = _hitPos;
    _startHitPos_o = _hitPos * w_to_interactor; //--> need Start hitPos ????!!!!
//########################################


    _scaleXonly =_hitNode == scaleXSphereGeode;// (_hitNode == scaleXaxisGeode) | (_hitNode == scaleXSphereGeode); // OR operation
    _scaleYonly =_hitNode == scaleYSphereGeode;// (_hitNode == scaleYaxisGeode) | (_hitNode == scaleYSphereGeode);
    _scaleZonly =_hitNode == scaleZSphereGeode;// (_hitNode == scaleZaxisGeode) | (_hitNode == scaleZSphereGeode);
    _scaleAll =_hitNode == sphereGeode;

    //create Cylinders which show the scale
    osg::ShapeDrawable *tempCylDrawable;
    auto cyl = new osg::Cylinder(osg::Vec3(0,0,0), 0.3, ArrowLength); 
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    tempCylDrawable = new osg::ShapeDrawable(cyl, hint);    
    tempCylDrawable->setColor(osg::Vec4(0.7,0.5,0.5,1));

    osg::Geode* tmpGeode = new osg::Geode();
    tmpGeode->addDrawable(tempCylDrawable);

    
    if(_scaleXonly)
    {
        tempxAxisTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(90.), 0, 1, 0)*osg::Matrix::translate(osg::Vec3(ArrowLength, 0, 0)*0.5));
        tempxAxisTransform->addChild(tmpGeode);
        _startxAxisMatrix = tempxAxisTransform->getMatrix();
    }
    else if(_scaleYonly)
    {
        tempyAxisTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(-90.), 1, 0, 0)*osg::Matrix::translate(osg::Vec3(0, ArrowLength, 0)*0.5));
        tempyAxisTransform->addChild(tmpGeode);
        _startyAxisMatrix = tempyAxisTransform->getMatrix();
    }
    else if(_scaleZonly)
    {
        tempzAxisTransform->setMatrix(osg::Matrix::translate(osg::Vec3(0, 0, ArrowLength)*0.5));;
        tempzAxisTransform->addChild(tmpGeode);
        _startzAxisMatrix = tempzAxisTransform->getMatrix();
    }
    else if(_scaleAll) // allow scale in all directions
    {
        tempxAxisTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(90.), 0, 1, 0)*osg::Matrix::translate(osg::Vec3(ArrowLength, 0, 0)*0.5));
        tempyAxisTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(-90.), 1, 0, 0)*osg::Matrix::translate(osg::Vec3(0, ArrowLength, 0)*0.5));
        tempzAxisTransform->setMatrix(osg::Matrix::translate(osg::Vec3(0, 0, ArrowLength)*0.5));

        tempxAxisTransform->addChild(tmpGeode);
        tempyAxisTransform->addChild(tmpGeode);
        tempzAxisTransform->addChild(tmpGeode);
        _startxAxisMatrix = tempxAxisTransform->getMatrix();
        _startyAxisMatrix = tempyAxisTransform->getMatrix(); 
        _startzAxisMatrix = tempzAxisTransform->getMatrix();
    }
   
    /* wie setze ich das hier um, brauch man das ? ###################################
    if (!_rotateOnly && !_translateOnly)
    {
        _translateOnly = is2D();
    }
    */

}

osg::Matrix coVR3DScaleGizmo::getMoveMatrix_o() const
{
    osg::Matrix scale;
    if(_scaleXonly)
        scale = osg::Matrix::scale(_scale,1,1);
    else if(_scaleYonly)
        scale = osg::Matrix::scale(1,_scale,1);
    else if(_scaleZonly)
        scale = osg::Matrix::scale(1,1,_scale);
    else
       scale = osg::Matrix::scale(_scale,_scale,_scale);

    return scale;
}

float coVR3DScaleGizmo::calcScale(osg::Vec3 axis)
{
    osg::Vec3 lp0_o, lp1_o, pointerDir_o;
    calculatePointerDirection_o(lp0_o, lp1_o, pointerDir_o);

    // calc Hit Pos on Gizmo
    osg::Matrix interactor_to_w = getMatrix();
    osg::Matrix w_to_interactor;
    w_to_interactor.invert(interactor_to_w);
    osg::Vec3 hitPos_o = _hitPos * w_to_interactor; //--> need Start hitPos ????!!!!

    // calc actual distance from pointer(transfered point on oxis) to center of gizmo
    osg::Vec3 pointOnAxis = calculatePointOfShortestDistance(lp0_o, lp1_o, axis); //Hit point on Xaxis
    osg::Vec3 pointOnAxis_o = pointOnAxis * w_to_interactor; // = distance to hitPoint
    osg::Vec3 d_pointer_center = hitPos_o + pointOnAxis_o; // = distance to center of gizmo
        

    float scale;
    int index;
    if (axis == osg::X_AXIS)
        index = 0;
    else if (axis == osg::Y_AXIS)
        index = 1;
    else if (axis == osg::Z_AXIS)
        index = 2;

    if(d_pointer_center.length() >= _startHitPos_o.length() && pointOnAxis_o[index] >= 0.0) 
        scale = 1 + pointOnAxis_o.length();
    else if(d_pointer_center.length() < _startHitPos_o.length() && d_pointer_center[index] >= 0)
        scale = d_pointer_center.length() / _startHitPos_o.length();
    else if(d_pointer_center.length() < _startHitPos_o.length() && d_pointer_center[index] < 0)
        scale = -d_pointer_center.length() / _startHitPos_o.length();
    else
        scale = (-1 + (2*hitPos_o.length() - pointOnAxis_o.length()));

    return scale;
}

void coVR3DScaleGizmo::doInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DScaleGizmo::move\n");
    

    // forbid translation in y-direction if traverseInteractors is on ############## wozu brauche ich das ? 
    // if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRConfig::instance()->useWiiNavigationVisenso())
    // {
        // osg::Vec3 trans = currHandMat.getTrans();
        // trans[1] = _oldHandMat.getTrans()[1];
        // currHandMat.setTrans(trans);
    // }

    // get hand pos in object coords
    osg::Vec3 lp0_o, lp1_o, pointerDir_o;
    calculatePointerDirection_o(lp0_o, lp1_o, pointerDir_o);
    
    
    osg::Matrix interactor_to_w = getMatrix();
    osg::Matrix w_to_interactor;
    w_to_interactor.invert(interactor_to_w);
    osg::Vec3 hitPos_o = _hitPos * w_to_interactor;
    float d_hitPos_center = hitPos_o.length(); 
    
    if(_scaleXonly)
        _scale = calcScale(osg::X_AXIS);
    else if(_scaleYonly)
        _scale = calcScale(osg::Y_AXIS);
    else if(_scaleZonly)
        _scale = calcScale(osg::Z_AXIS);
    else if(_scaleAll) // allow scale in all directions
    {
        osg::Matrix w_to_o = cover->getInvBaseMat();
        osg::Matrix currHandMat = getPointerMat();
        osg::Matrix currHandMat_o = currHandMat * w_to_o;
        osg::Vec3 currHandPos_o = currHandMat_o.getTrans();  
        osg::Vec3 interPos = currHandPos_o + pointerDir_o* _distance + _diff; //without the diff we measure distance to center of gizmo , but this is now not the interPos
        float d_inter_center = (interPos - getMatrix().getTrans()).length();
        //_scale = 1 + d_inter_center;
        _scale = d_inter_center;
    }
}

osg::Vec3 coVR3DScaleGizmo::calculatePointOfShortestDistance(const osg::Vec3& lp0, const osg::Vec3& lp1, osg::Vec3 axis_o) const
{
    osg::Vec3 newPos_o, pointLine1, pointLine2;
    osg::Matrix interactor_to_w = getMatrix();
    osg::Vec3 startInterMat_o = osg::Matrix::transform3x3(_startInterMat_w.getTrans(), interactor_to_w.inverse(interactor_to_w));
    _helperLine->update(axis_o.operator*(-50.0f*getScale())*getMatrix(),  axis_o.operator*(50.0f*getScale()) *getMatrix()); //the length of the helper line doesn't matter, these values are just taken for visualization
    // if(_helperLine->getPointsOfShortestDistance(lp0, lp1, pointLine1, pointLine2))  what happens if lines are parallel ? 
    // {
        _helperLine->getPointsOfShortestDistance(lp0, lp1, pointLine1, pointLine2);
        newPos_o =osg::Matrix::transform3x3(pointLine1 + _diff,interactor_to_w.inverse(interactor_to_w));

        //newPos = pointLine1 + _diff;
        if(axis_o == osg::X_AXIS)
        {
            //_helperLine->setColor(_red);
            newPos_o.z() = startInterMat_o.z();
            newPos_o.y() = startInterMat_o.y();   
        }
        else if(axis_o == osg::Y_AXIS)
        {
            //_helperLine->setColor(_green);
            newPos_o.z() = startInterMat_o.z();
            newPos_o.x() = startInterMat_o.x();
        }
        else if(axis_o == osg::Z_AXIS)
        {
            //_helperLine->setColor(_blue);
            newPos_o.x() = startInterMat_o.x();
            newPos_o.y() = startInterMat_o.y();
        }

        //_helperLine->show();

    // }
    // else
        // newPos = _oldInteractorXformMat_o.getTrans();
    osg::Vec3 newPos_w = osg::Matrix::transform3x3(newPos_o,interactor_to_w);

    return newPos_w;
}
void coVR3DScaleGizmo::stopInteraction()
{
    tempxAxisTransform->removeChildren(0,tempxAxisTransform->getNumChildren());
    tempyAxisTransform->removeChildren(0,tempyAxisTransform->getNumChildren());
    tempzAxisTransform->removeChildren(0,tempzAxisTransform->getNumChildren());

    coVR3DGizmoType::stopInteraction();

}

void coVR3DScaleGizmo::updateTransform(osg::Matrix m)
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coVR3DScaleGizmo:setMatrix\n");
    _interMat_o = m;
    moveTransform->setMatrix(m);
    if (m_sharedState)
    {
        if (auto st = static_cast<SharedMatrix *>(m_sharedState.get()))
        {
            *st = m;
        }
    }
}
/*
void coVR3DScaleGizmo::setShared(bool shared)
{
    if (shared)
    {
        if (!m_sharedState)
        {
            m_sharedState.reset(new SharedMatrix("interactor." + std::string(_interactorName), _oldInteractorXformMat_o));
            m_sharedState->setUpdateFunction([this]() {
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
                updateTransform(interactorXformMat_o);
            });
        }
    }
    else
    {
        m_sharedState.reset(nullptr);
    }
}

*/

int coVR3DScaleGizmo::hit(vrui::vruiHit *hit)
{    
    osg::Node* oldHitNode;
                                                                                                                                                                  
    if(hit)
        oldHitNode = _hitNode.get();    

    int returnValue = coVR3DGizmoType::hit(hit);
    
    // set color of Sphere if axis is selected and vice versa
    if(_hitNode == scaleXaxisGeode)
        scaleXSphereGeode->setStateSet(_hitNode->getStateSet());
    else if(_hitNode == scaleXSphereGeode)
        scaleXaxisGeode->setStateSet(_hitNode->getStateSet());
    else if(_hitNode == scaleYaxisGeode)
        scaleYSphereGeode->setStateSet(_hitNode->getStateSet());
    else if (_hitNode == scaleYSphereGeode)
        scaleYaxisGeode->setStateSet(_hitNode->getStateSet());
    else if(_hitNode == scaleZaxisGeode)
        scaleZSphereGeode->setStateSet(_hitNode->getStateSet());
    else if (_hitNode == scaleZSphereGeode)
        scaleZaxisGeode->setStateSet(_hitNode->getStateSet());

    // reset color of axis / sphere if you move from one hit node directly to another hit node    
    osg::Node* newHitNode = _hitNode;
    if(oldHitNode == scaleXaxisGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr ) 
        scaleXSphereGeode->setStateSet(NULL); 
    else if(oldHitNode == scaleXSphereGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr)  
        scaleXaxisGeode->setStateSet(NULL);
    else if(oldHitNode == scaleYaxisGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr ) 
        scaleYSphereGeode->setStateSet(NULL); 
    else if(oldHitNode == scaleYSphereGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr)  
        scaleYaxisGeode->setStateSet(NULL); 
    else if(oldHitNode == scaleZaxisGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr ) 
        scaleZSphereGeode->setStateSet(NULL); 
    else if(oldHitNode == scaleZSphereGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr)  
        scaleZaxisGeode->setStateSet(NULL); 

    return returnValue;

}

void coVR3DScaleGizmo::resetState()
{
    coVR3DGizmoType::resetState();
    
    scaleXaxisGeode->setStateSet(NULL);
    scaleYaxisGeode->setStateSet(NULL);
    scaleZaxisGeode->setStateSet(NULL);
    scaleXSphereGeode->setStateSet(NULL);
    scaleYSphereGeode->setStateSet(NULL);
    scaleZSphereGeode->setStateSet(NULL);

}