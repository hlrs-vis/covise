#include "coVR3DRotGizmo.h"

#include <OpenVRUI/osg/mathUtils.h>
#include <osg/MatrixTransform>
#include <cover/coVRNavigationManager.h>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <osg/io_utils>
#include <osgSim/SphereSegment>
#include <vrb/client/SharedState.h>

using namespace opencover;

coVR3DRotGizmo::coVR3DRotGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium, coVR3DGizmo* gizmoPointer)
    :coVR3DGizmoType(m, s, type, iconName, interactorName, priority, gizmoPointer)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR3DRotGizmo(%s)\n", interactorName);
    }

    createGeometry();

    updateTransform(m);

    
}

coVR3DRotGizmo::~coVR3DRotGizmo()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~coVR3DRotGizmo\n");
}

void coVR3DRotGizmo::createGeometry()
{
    osg::ShapeDrawable *sphereDrawable;
    osg::Vec3 origin(0, 0, 0);

    _axisTransform = new osg::MatrixTransform();
    _axisTransform->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    geometryNode = _axisTransform.get();
    scaleTransform->addChild(geometryNode.get());


    osg::Sphere *mySphere = new osg::Sphere(origin, 0.5);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereDrawable->setColor(_grey);
    _sphereGeode = new osg::Geode();
    _sphereGeode->addDrawable(sphereDrawable);
    _axisTransform->addChild(_sphereGeode.get());

    _xRotCylGroup = circlesFromCylinders(RotationAxis::X, 24, _red, _radius/4);
    _yRotCylGroup = circlesFromCylinders(RotationAxis::Y, 24, _green, _radius/4);
    _zRotCylGroup = circlesFromCylinders(RotationAxis::Z, 24, _blue, _radius/4);

    _axisTransform->addChild(_xRotCylGroup);
    _axisTransform->addChild(_yRotCylGroup);
    _axisTransform->addChild(_zRotCylGroup);
}

osg::Vec3Array* coVR3DRotGizmo::circleVerts(RotationAxis axis, int approx)
{
    const double angle( osg::PI * 2. / (double) approx );
    osg::Vec3Array* v = new osg::Vec3Array;
    int idx;

    for( idx=0; idx<approx; idx++)
    {
        double cosAngle = cos(idx*angle);
        double sinAngle = sin(idx*angle);
        double x(0.), y(0.), z(0.);
        switch (axis) {
            case RotationAxis::Z: 
                x = cosAngle*_radius;
                y = sinAngle*_radius;
                break;
            case RotationAxis::X:
                y = cosAngle*_radius;
                z = sinAngle*_radius;
                break;
            case RotationAxis::Y: 
                x = cosAngle*_radius;
                z = sinAngle*_radius;
                break;
        }
        v->push_back( osg::Vec3( x, y, z ) );
    }
    
    return v;
}

osg::Geode* coVR3DRotGizmo:: circles( RotationAxis axis, int approx, osg::Vec4 color )
{
    osg::Geode* geode = new osg::Geode;
    osg::LineWidth* lw = new osg::LineWidth( 4. );
    geode->getOrCreateStateSet()->setAttributeAndModes( lw,
    osg::StateAttribute::ON );


    osg::Geometry* geom = new osg::Geometry;
    osg::Vec3Array* v = circleVerts( axis, approx );
    geom->setVertexArray( v );

    osg::Vec4Array* c = new osg::Vec4Array;
    c->push_back( color );
    geom->setColorArray( c );
    geom->setColorBinding( osg::Geometry::BIND_OVERALL );
    geom->addPrimitiveSet( new osg::DrawArrays( GL_LINE_LOOP, 0, approx ) );

    geode->addDrawable( geom );
    return geode;
}

osg::Group* coVR3DRotGizmo:: circlesFromCylinders( RotationAxis axis, int approx, osg::Vec4 color, float cylLength )
{
    osg::Group* parent = new osg::Group;
    osg::ShapeDrawable *cylDrawable;
    auto cyl = new osg::Cylinder(osg::Vec3(0,0,0), 0.15, cylLength);
    cylDrawable = new osg::ShapeDrawable(cyl);
    cylDrawable->setColor(color);
    osg::Geode* geode = new osg::Geode;
    geode->addDrawable(cylDrawable);
    
    osg::Vec3Array* v = circleVerts(axis, approx );

    const double angle( 360.0 / (double) approx );
    double incrementAngle{0.0};

    for(osg::Vec3Array::iterator vitr = v->begin(); vitr != v->end(); ++vitr)
    {
        coCoord euler;
        euler.xyz = *vitr;
        if(axis == RotationAxis::Z)
            euler.hpr[1] = 90; 
    
        euler.hpr[(int)axis] = incrementAngle; 
        osg::Matrix matrix;
        euler.makeMat(matrix);

        osg::MatrixTransform *matrixTransform = new osg::MatrixTransform(matrix);
        matrixTransform->addChild(geode);
        parent->addChild(matrixTransform);
        if(axis == RotationAxis::X || axis == RotationAxis::Z )
            incrementAngle += angle;
        else if(axis == RotationAxis::Y  )
            incrementAngle -= angle;
    }

    return parent;
}



void coVR3DRotGizmo::startInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DRotGizmo::startInteraction\n");


    osg::Vec3 lp0_o, lp1_o, pointerDir_o;
    calculatePointerDirection_o(lp0_o, lp1_o, pointerDir_o);
    
    coVRIntersectionInteractor::startInteraction();

    _startInterMat_w = getMatrix();
    _oldInterMat_o = _startInterMat_w;
    _startHandMat = getPointerMat();

    _rotateZonly = rotateAroundSpecificAxis(_zRotCylGroup.get());
    _rotateYonly = rotateAroundSpecificAxis(_yRotCylGroup.get());
    _rotateXonly = rotateAroundSpecificAxis(_xRotCylGroup.get());

    osg::Matrix rotateOnly = _startInterMat_w;
    rotateOnly.setTrans(osg::Vec3(0,0,0));
    if(_rotateZonly)
    {
        _helperPlane->update(osg::Z_AXIS * rotateOnly,_startInterMat_w.getTrans());
        closestDistanceLineCircle(lp0_o, lp1_o, osg::Z_AXIS * rotateOnly, _startPointOnCircle);

        _helperLine->setColor(_blue);
        _helperLine->update(osg::Vec3(0,0,-_radius*3*getScale())*getMatrix(),osg::Vec3(0,0,_radius*3*getScale())*getMatrix());
        _helperLine->show();
    }
    else if(_rotateYonly)
    {
        _helperPlane->update(osg::Y_AXIS * rotateOnly,_startInterMat_w.getTrans());
        closestDistanceLineCircle(lp0_o, lp1_o, osg::Y_AXIS* rotateOnly, _startPointOnCircle);

        _helperLine->setColor(_green);
        _helperLine->update(osg::Vec3(0,-_radius*3*getScale(),0)*getMatrix(),osg::Vec3(0,_radius*3*getScale(),0)*getMatrix());
        _helperLine->show();
    }
    else if(_rotateXonly)
    {
        _helperPlane->update(osg::X_AXIS * rotateOnly,_startInterMat_w.getTrans());
        closestDistanceLineCircle(lp0_o, lp1_o, osg::X_AXIS* rotateOnly, _startPointOnCircle);

        _helperLine->setColor(_red);
        _helperLine->update(osg::Vec3(-_radius*3*getScale(),0,0)*getMatrix(),osg::Vec3(_radius*3*getScale(),0,0)*getMatrix());
        _helperLine->show();
    }
   

    // if (!_rotateOnly && !_translateOnly)
    // {
    //     _translateOnly = is2D();
    // }


}
//check if hitNode is child of specific rotation group
bool coVR3DRotGizmo::rotateAroundSpecificAxis(osg::Group *group) const
{
    bool foundNode{false};
    int numChildren = group->getNumChildren();

    for(int i{0}; i < numChildren; i++)
    {
         auto node = group->getChild(i);
         if(_hitNode->getParent(0) == node)
            foundNode = true;
    }
    return foundNode;
}


void coVR3DRotGizmo::doInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DRotGizmo::rot\n");

    osg::Vec3 origin(0, 0, 0);
    osg::Vec3 yaxis(0, 1, 0);
    osg::Matrix o_to_w = cover->getBaseMat();
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat = getPointerMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;

    // forbid translation in y-direction if traverseInteractors is on --> why do we need this ???? ###############################
    // if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRConfig::instance()->useWiiNavigationVisenso())
    // {
    //     osg::Vec3 trans = currHandMat.getTrans();
    //     trans[1] = _oldHandMat.getTrans()[1];
    //     currHandMat.setTrans(trans);
    // }
    osg::Vec3 lp0_o, lp1_o, pointerDir_o;
    calculatePointerDirection_o(lp0_o, lp1_o, pointerDir_o);

    // translate from interactor to hand and back
    osg::Matrix transToHand_o, revTransToHand_o;

    transToHand_o.makeTranslate(currHandMat_o.getTrans() - _oldInterMat_o.getTrans());
    revTransToHand_o.makeTranslate(_oldInterMat_o.getTrans() - currHandMat_o.getTrans());

    osg::Matrix relHandMoveMat_o = _invOldHandMat_o * currHandMat_o;
    //std::cout << "rel Hand Movement:" <<relHandMoveMat_o.getTrans() <<"..."<< std::endl;
    osg::Matrix newInteractorMatrix = _oldInterMat_o;
    if (_rotateZonly)
    {
        if(is2D())
            newInteractorMatrix = calcRotation2D(lp0_o, lp1_o, osg::Z_AXIS);
        else
            newInteractorMatrix = calcRotation3D(osg::Z_AXIS);
    }
    else if(_rotateYonly)
    {
        if(is2D())
            newInteractorMatrix = calcRotation2D(lp0_o, lp1_o, osg::Y_AXIS);
        else
            newInteractorMatrix = calcRotation3D(osg::Y_AXIS);
    }
    else if(_rotateXonly)
    {
        if(is2D())
            newInteractorMatrix = calcRotation2D(lp0_o, lp1_o, osg::X_AXIS);
        else
            newInteractorMatrix = calcRotation3D(osg::X_AXIS);
    }
    else if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors)
    {
        // move old mat to hand position, apply rel hand movement and move it back to
        newInteractorMatrix = _oldInterMat_o * transToHand_o * relHandMoveMat_o * revTransToHand_o;
    }
    else
    {
        //if(!is2D())
            newInteractorMatrix = _oldInterMat_o * relHandMoveMat_o; // apply rel hand movement
    }

    // save old transformation
    _oldInterMat_o = newInteractorMatrix;

    _oldHandMat = currHandMat; // save current hand for rotation start
    _invOldHandMat_o.invert(currHandMat_o);

    if (cover->restrictOn())
    {
        // restrict to visible scene
        osg::Vec3 pos_o, restrictedPos_o;
        pos_o = newInteractorMatrix.getTrans();
        restrictedPos_o = restrictToVisibleScene(pos_o);
        newInteractorMatrix.setTrans(restrictedPos_o);
    }

    if (coVRNavigationManager::instance()->isSnapping())
    {
        if (coVRNavigationManager::instance()->isDegreeSnapping())
        {
            // snap orientation
            snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &newInteractorMatrix);
        }
        else
        {
            // snap orientation to 45 degree
            snapTo45Degrees(&newInteractorMatrix);
        }
    }


    // and now we apply it
     updateTransform(newInteractorMatrix);

}

//Math: https://nelari.us/post/gizmos/
osg::Matrix coVR3DRotGizmo::calcRotation2D(const osg::Vec3& lp0_o, const osg::Vec3& lp1_o, osg::Vec3 rotationAxis)
{
    osg::Vec3 pointOnCircle, normal;
    osg::Matrix rotationOnly = _startInterMat_w;
    rotationOnly.setTrans(osg::Vec3(0,0,0));
    normal = rotationAxis * rotationOnly;
    normal.normalize();

    closestDistanceLineCircle(lp0_o, lp1_o, normal, pointOnCircle);

    // create two direction vectors
    osg::Vec3 dir1 = pointOnCircle -getMatrix().getTrans(); 
    osg::Vec3 dir2 = _startPointOnCircle -getMatrix().getTrans(); 
    dir1.normalize();
    dir2.normalize();
    
    osg::Matrix totalRotation; // the rotation from start of interaction to current curser position
    //calculate the angle between two direction vectors using the normal as a reference
    totalRotation.makeRotate(osg::DegreesToRadians(vecAngle360(dir1, dir2, -normal)), normal);
    
    
    osg::Matrix startRotation = _startInterMat_w;
    startRotation.setTrans(osg::Vec3(0,0,0));
    
    return startRotation * totalRotation * osg::Matrix::translate(_startInterMat_w.getTrans());
}

osg::Matrix coVR3DRotGizmo::calcRotation3D(osg::Vec3 rotationAxis)
{
    
    osg::Matrix interMatrix;
    coCoord startEuler = _startHandMat;
    coCoord currentEuler = getPointerMat();

    float angle = currentEuler.hpr[2] - startEuler.hpr[2];
    std::cout <<"angle: " <<angle <<" ..."<<std::endl;

    osg::Matrix rotate;
    rotate.makeRotate(osg::DegreesToRadians(angle), rotationAxis *getMatrix());
    interMatrix = _startInterMat_w*rotate;  

    return interMatrix; 
}

//Math: https://nelari.us/post/gizmos/
float coVR3DRotGizmo::closestDistanceLineCircle(const osg::Vec3& lp0, const osg::Vec3& lp1,const osg::Vec3 rotationAxis, osg::Vec3 &pointOnCircle) const
{
    osg::Vec3 isectPoint;
    _helperPlane->update(rotationAxis, _startInterMat_w.getTrans());

    // get the ray's intersection point on the plane which contains the circle
    bool intersect = _helperPlane->getLineIntersectionPoint( lp0, lp1, isectPoint);
  
    // project that point on to the circle's circumference
    osg::Vec3 center = getMatrix().getTrans();
    osg::Vec3 diff = isectPoint-center;
    diff.normalize();
    pointOnCircle = center + diff.operator*(_radius);
    float distance = (pointOnCircle - isectPoint).length();
    return distance; 
}

//function that takes two input vectors v1 and v2, and a vector n that is not in the plane of v1 & v2. 
//Here n is used to determine the "direction" of the angle between v1 and v2 in a right-hand-rule sense. 
//I.e., cross(n,v1) would point in the "positive" direction of the angle starting from v1
// math is from here: https://itectec.com/matlab/matlab-angle-betwen-two-3d-vectors-in-the-range-0-360-degree/
double coVR3DRotGizmo::vecAngle360(const osg::Vec3 vec1, const osg::Vec3 &vec2, const osg::Vec3& refVec)
{
    osg::Vec3 cross = vec1^vec2;
    int sign;

    if(cross*refVec >= 0)
        sign = 1;
    else if (cross*refVec <0)
        sign = -1;
    
    double angle = osg::RadiansToDegrees(std::atan2(sign * cross.length(), vec1*vec2));
    return angle;
}

void coVR3DRotGizmo::stopInteraction() 
{
    _helperLine->hide();
    coVR3DGizmoType::stopInteraction();
}

