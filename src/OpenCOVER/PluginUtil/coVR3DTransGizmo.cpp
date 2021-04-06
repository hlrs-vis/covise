#include "coVR3DTransGizmo.h"
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


using namespace opencover;

coVR3DTransGizmo::coVR3DTransGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium,coVR3DGizmo* gizmoPointer)
    : coVR3DGizmoType(m, s, type, iconName, interactorName, priority, gizmoPointer) 
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR3DTransGizmo(%s)\n", interactorName);
    }

    createGeometry();

}

coVR3DTransGizmo::~coVR3DTransGizmo()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~coVR3DTransGizmo\n");

}

void
coVR3DTransGizmo::createGeometry()
{
   if (cover->debugLevel(4))
        fprintf(stderr, "\ncoVR3DTransGizmo::createGeometry\n");


    osg::ShapeDrawable *sphereDrawable, *xCylDrawable, *yCylDrawable, *zCylDrawable, *xConeDrawable, *yConeDrawable, *zConeDrawable, *xyPlaneDrawable,*xzPlaneDrawable,*yzPlaneDrawable;

    osg::Vec3 origin(0, 0, 0), px(1, 0, 0), py(0, 1, 0), pz(0, 0, 1);
    osg::Vec3 yaxis(0, 1, 0);

    axisTransform = new osg::MatrixTransform();
    axisTransform->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    geometryNode = axisTransform.get();
    scaleTransform->addChild(geometryNode.get());

    osg::Sphere *mySphere = new osg::Sphere(origin, 0.5);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereDrawable->setColor(_grey);
    sphereGeode = new osg::Geode();
    sphereGeode->addDrawable(sphereDrawable);
    axisTransform->addChild(sphereGeode.get());

    //create axis
    auto cyl = new osg::Cylinder(origin, 0.15, _arrowLength);
    xCylDrawable = new osg::ShapeDrawable(cyl, hint);
    yCylDrawable = new osg::ShapeDrawable(cyl, hint);
    zCylDrawable = new osg::ShapeDrawable(cyl, hint);

    xCylDrawable->setColor(_red);
    yCylDrawable->setColor(_green);
    zCylDrawable->setColor(_blue);

    xAxisTransform = new osg::MatrixTransform();
    xAxisTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(90.), 0, 1, 0)*osg::Matrix::translate(osg::Vec3(_arrowLength, 0, 0)*0.5));
    yAxisTransform = new osg::MatrixTransform();
    yAxisTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(-90.), 1, 0, 0)*osg::Matrix::translate(osg::Vec3(0, _arrowLength, 0)*0.5));
    zAxisTransform = new osg::MatrixTransform();
    zAxisTransform->setMatrix(osg::Matrix::translate(osg::Vec3(0, 0, _arrowLength)*0.5));

    translateXaxisGeode = new osg::Geode;
    translateXaxisGeode->addDrawable(xCylDrawable);
    xAxisTransform->addChild(translateXaxisGeode);
    translateYaxisGeode = new osg::Geode;
    translateYaxisGeode->addDrawable(yCylDrawable);
    yAxisTransform->addChild(translateYaxisGeode);
    translateZaxisGeode = new osg::Geode;
    translateZaxisGeode->addDrawable(zCylDrawable);
    zAxisTransform->addChild(translateZaxisGeode);

    axisTransform->addChild(xAxisTransform);
    axisTransform->addChild(yAxisTransform);
    axisTransform->addChild(zAxisTransform);

    // create cones
    osg::Cone *myCone = new osg::Cone(origin, 0.5, 2.0);
    xConeDrawable = new osg::ShapeDrawable(myCone);
    yConeDrawable = new osg::ShapeDrawable(myCone);
    zConeDrawable = new osg::ShapeDrawable(myCone);

    xConeDrawable->setColor(_red);
    yConeDrawable->setColor(_green);
    zConeDrawable->setColor(_blue);

    xConeTransform = new osg::MatrixTransform();
    xConeTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(90.), 0, 1, 0)*osg::Matrix::translate(osg::Vec3(_arrowLength, 0, 0)));
    yConeTransform = new osg::MatrixTransform();
    yConeTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(-90.), 1, 0, 0)*osg::Matrix::translate(osg::Vec3(0, _arrowLength, 0)));
    zConeTransform = new osg::MatrixTransform();
    zConeTransform->setMatrix(osg::Matrix::translate(osg::Vec3(0, 0, _arrowLength)));

    translateXconeGeode = new osg::Geode;
    translateXconeGeode->addDrawable(xConeDrawable);
    xConeTransform->addChild(translateXconeGeode);
    translateYconeGeode = new osg::Geode;
    translateYconeGeode->addDrawable(yConeDrawable);
    yConeTransform->addChild(translateYconeGeode);
    translateZconeGeode = new osg::Geode;
    translateZconeGeode->addDrawable(zConeDrawable);
    zConeTransform->addChild(translateZconeGeode);

    axisTransform->addChild(xConeTransform);
    axisTransform->addChild(yConeTransform);
    axisTransform->addChild(zConeTransform);

    //create planes
    auto plane = new osg::Box(origin,_arrowLength/3, 0.1, _arrowLength/3);
    xzPlaneDrawable = new osg::ShapeDrawable(plane, hint);
    xyPlaneDrawable = new osg::ShapeDrawable(plane, hint);
    yzPlaneDrawable = new osg::ShapeDrawable(plane, hint);

    xyPlaneDrawable->setColor(_grey);
    xzPlaneDrawable->setColor(_grey);
    yzPlaneDrawable->setColor(_grey);

    xzPlaneTransform = new osg::MatrixTransform();
    xzPlaneTransform->setMatrix(osg::Matrix::translate(osg::Vec3(_arrowLength/3, 0, _arrowLength/3)));
    xyPlaneTransform = new osg::MatrixTransform();
    xyPlaneTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(-90.), 1, 0, 0)*osg::Matrix::translate(osg::Vec3(_arrowLength/3, _arrowLength/3, 0)));
    yzPlaneTransform = new osg::MatrixTransform();
    yzPlaneTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(-90.), 0, 0, 1)*osg::Matrix::translate(osg::Vec3(0, _arrowLength/3, _arrowLength/3)));

    translateXZplaneGeode = new osg::Geode;
    translateXZplaneGeode->addDrawable(xzPlaneDrawable);
    xzPlaneTransform->addChild(translateXZplaneGeode);
    translateXYplaneGeode = new osg::Geode;
    translateXYplaneGeode->addDrawable(xyPlaneDrawable);
    xyPlaneTransform->addChild(translateXYplaneGeode);
    translateYZplaneGeode = new osg::Geode;
    translateYZplaneGeode->addDrawable(yzPlaneDrawable);
    yzPlaneTransform->addChild(translateYZplaneGeode);

    axisTransform->addChild(xzPlaneTransform);
    axisTransform->addChild(xyPlaneTransform);
    axisTransform->addChild(yzPlaneTransform);

}

void coVR3DTransGizmo::startInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DTransGizmo::startInteraction\n");

    _translateXonly = (_hitNode == translateXaxisGeode) | (_hitNode == translateXconeGeode);
    _translateYonly = (_hitNode == translateYaxisGeode) | (_hitNode == translateYconeGeode);
    _translateZonly = (_hitNode == translateZaxisGeode) | (_hitNode == translateZconeGeode);
    _translateXYonly = _hitNode == translateXYplaneGeode;
    _translateXZonly = _hitNode == translateXZplaneGeode;
    _translateYZonly = _hitNode == translateYZplaneGeode;

    coVR3DGizmoType::startInteraction();
    osg::Matrix interactor_to_w = getMatrix();

}
void coVR3DTransGizmo::doInteraction()
{

    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DTransGizmo::move\n");
    
    osg::Vec3 lp0_o, lp1_o, pointerDir_o;
    calculatePointerDirection_o(lp0_o, lp1_o, pointerDir_o);

    if(_translateXonly)
        _interMat_o.setTrans(calculatePointOfShortestDistance(lp0_o, lp1_o, osg::X_AXIS)); 

    else if(_translateYonly)
        _interMat_o.setTrans(calculatePointOfShortestDistance(lp0_o, lp1_o, osg::Y_AXIS)); 

    else if(_translateZonly)
        _interMat_o.setTrans(calculatePointOfShortestDistance(lp0_o, lp1_o, osg::Z_AXIS)); 

    else if(_translateXYonly)
        _interMat_o.setTrans(calcPlaneLineIntersection(lp0_o, lp1_o, osg::Z_AXIS)); 

    else if(_translateXZonly)
        _interMat_o.setTrans(calcPlaneLineIntersection(lp0_o, lp1_o, osg::Y_AXIS)); 

    else if(_translateYZonly)
        _interMat_o.setTrans(calcPlaneLineIntersection(lp0_o, lp1_o, osg::X_AXIS)); 

    else // allow translation in all directions 
    {
        osg::Matrix w_to_o = cover->getInvBaseMat();
        osg::Matrix currHandMat = getPointerMat();
        osg::Matrix currHandMat_o = currHandMat * w_to_o;
        osg::Vec3 currHandPos_o = currHandMat_o.getTrans();  
        osg::Vec3 interPos = currHandPos_o + pointerDir_o * _distance + _diff;

        _interMat_o.setTrans(interPos); 
    }


    if (cover->restrictOn())
    {
        // restrict to visible scene
        osg::Vec3 pos_o, restrictedPos_o;
        pos_o = _interMat_o.getTrans();
        restrictedPos_o = restrictToVisibleScene(pos_o);
        _interMat_o.setTrans(restrictedPos_o);
    }

    // and now we apply it
    updateTransform(_interMat_o);
}


osg::Vec3 coVR3DTransGizmo::calculatePointOfShortestDistance(const osg::Vec3& lp0, const osg::Vec3& lp1, osg::Vec3 axis_o) const
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


void coVR3DTransGizmo::stopInteraction() 
{
    //_helperLine->hide();
    coVR3DGizmoType::stopInteraction();
}

int coVR3DTransGizmo::hit(vrui::vruiHit *hit)
{    
    osg::Node* oldHitNode;
                                                                                                                                                                  
    if(hit)
        oldHitNode = _hitNode.get();    

    int returnValue = coVR3DGizmoType::hit(hit);
    
    // set color of cone if axis is selected and vice versa
    if(_hitNode == translateXaxisGeode)
        translateXconeGeode->setStateSet(_hitNode->getStateSet());
    else if(_hitNode == translateXconeGeode)
        translateXaxisGeode->setStateSet(_hitNode->getStateSet());
    else if(_hitNode == translateYaxisGeode)
        translateYconeGeode->setStateSet(_hitNode->getStateSet());
    else if (_hitNode == translateYconeGeode)
        translateYaxisGeode->setStateSet(_hitNode->getStateSet());
    else if(_hitNode == translateZaxisGeode)
        translateZconeGeode->setStateSet(_hitNode->getStateSet());
    else if (_hitNode == translateZconeGeode)
        translateZaxisGeode->setStateSet(_hitNode->getStateSet());

    // reset color of axis / cone if you move from one hit node directly to another hit node    
    osg::Node* newHitNode = _hitNode;
    if(oldHitNode == translateXaxisGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr ) 
        translateXconeGeode->setStateSet(NULL); 
    else if(oldHitNode == translateXconeGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr)  
        translateXaxisGeode->setStateSet(NULL);
    else if(oldHitNode == translateYaxisGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr ) 
        translateYconeGeode->setStateSet(NULL); 
    else if(oldHitNode == translateYconeGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr)  
        translateYaxisGeode->setStateSet(NULL); 
    else if(oldHitNode == translateZaxisGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr ) 
        translateZconeGeode->setStateSet(NULL); 
    else if(oldHitNode == translateZconeGeode && oldHitNode != newHitNode && _interactionHitNode == nullptr)  
        translateZaxisGeode->setStateSet(NULL); 

    return returnValue;

}

void coVR3DTransGizmo::resetState()
{
    coVR3DGizmoType::resetState();
    
    translateXaxisGeode->setStateSet(NULL);
    translateYaxisGeode->setStateSet(NULL);
    translateZaxisGeode->setStateSet(NULL);
    translateXconeGeode->setStateSet(NULL);
    translateYconeGeode->setStateSet(NULL);
    translateZconeGeode->setStateSet(NULL);

}







