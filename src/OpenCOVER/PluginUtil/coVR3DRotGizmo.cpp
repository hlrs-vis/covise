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

    if(_rotateZonly)
    {
        _helperPlane->update(osg::Z_AXIS * getMatrix(),osg::Vec3(0,0,0)*getMatrix());
        closestDistanceLineCircle(lp0_o, lp1_o, osg::Z_AXIS * getMatrix(), _closestStartPoint_o);

        _helperLine->setColor(_blue);
        _helperLine->update(osg::Vec3(0,0,-_radius*3*getScale())*getMatrix(),osg::Vec3(0,0,_radius*3*getScale())*getMatrix());
        _helperLine->show();
    }
    else if(_rotateYonly)
    {
        _helperPlane->update(osg::Y_AXIS * getMatrix(),osg::Vec3(0,0,0));
        closestDistanceLineCircle(lp0_o, lp1_o, osg::Y_AXIS* getMatrix(), _closestStartPoint_o);

        _helperLine->setColor(_green);
        _helperLine->update(osg::Vec3(0,-_radius*3*getScale(),0)*getMatrix(),osg::Vec3(0,_radius*3*getScale(),0)*getMatrix());
        _helperLine->show();
    }
    else if(_rotateXonly)
    {
        _helperPlane->update(osg::X_AXIS * getMatrix(),osg::Vec3(0,0,0));
        closestDistanceLineCircle(lp0_o, lp1_o, osg::X_AXIS* getMatrix(), _closestStartPoint_o);

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

float coVR3DRotGizmo::closestDistanceLineCircle(const osg::Vec3& lp0, const osg::Vec3& lp1,const osg::Vec3 rotationAxis, osg::Vec3 &closestPoint) const
{
    osg::Vec3 isectPoint;
    _helperPlane->update(rotationAxis, getMatrix().getTrans());
    bool intersect = _helperPlane->getLineIntersectionPoint( lp0, lp1, isectPoint);
    //newPos  = isectPoint + _diff;
    osg::Vec3 normalized = isectPoint - getMatrix().getTrans();
    //std::cout <<"isect Point" << isectPoint <<std::endl;
    normalized.normalize();
    closestPoint = getMatrix().getTrans() +  normalized.operator*(_radius); // veränder sich der Radius beim Zoomen ?
    // std::cout <<" closest Point" <<closestPoint<<"..." <<std::endl;
    return 1.0f; // FIXME: if no intersection!! 
}

//math is from here: https://itectec.com/matlab/matlab-angle-betwen-two-3d-vectors-in-the-range-0-360-degree/
double coVR3DRotGizmo::vecAngle360(const osg::Vec3 vec1, const osg::Vec3 &vec2, const osg::Vec3& refVec)
{
    osg::Vec3 cross = vec1^vec2;
    int sign;

    if(cross*refVec >= 0)
        sign = 1;
    else if (cross*refVec <0)
        sign = -1;
    
    double angle = osg::RadiansToDegrees(std::atan2(sign * cross.length(), vec1*vec2));
    // std::cout <<"angle: " <<angle << ".." << std::endl;
    return angle;
}

/*
osg::Matrix coVR3DRotGizmo::calcRotation2D(osg::Vec3 rotationAxis, osg::Vec3 cylinderDirectionVector)
{
    osg::Matrix interactorXformMat_o; 
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat = getPointerMat();
    osg::Vec3 origin{0, 0, 0};
    osg::Vec3 yaxis(0, 1, 0);
    osg::Matrix i_to_o = scaleTransform->getMatrix()*moveTransform->getMatrix();
    osg::Matrix o_to_i = osg::Matrix::inverse(i_to_o);
    osg::Vec3 hand_i = origin * currHandMat * w_to_o * o_to_i;
    osg::Vec3 pos = hand_i;
    osg::Vec3 dir = yaxis * currHandMat * w_to_o * o_to_i;
    dir -= pos;
    dir.normalize();
    // std::cerr << "pos: " << pos << ", dir: " << dir << std::endl;
    double R = _diff.length() / getScale();
    double a = dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2];
    double b = 2.*(dir[0]*pos[0] + dir[1]*pos[1] + dir[2]*pos[2]);
    double c = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2] - R*R;
    double D = b*b-4*a*c;
    // std::cerr << "scale=" << getScale() << ", a=" << a << ", b=" << b << ", c=" << c << ", disc=" << D << std::endl;
    double t = -1.;
    if (D >= 0)
    {
        double t1 = 0.5*(-b-sqrt(D))/a;
        double t2 = 0.5*(-b+sqrt(D))/a;
        if (t1 < 0)
        {
            t = t2;
        }
        else if (is2D())
        {
            t = t1;
        }
        else
        {
            double old = _distance / getScale();
            if (std::abs(old-t1) < std::abs(old-t2))
                t = t1;
            else
                t = t2;
        }
        // std::cerr << "solution: t1=" << t1 << ", t2=" << t2 << ", t=" << t << std::endl;
        // // osg::Vec3 v1 = pos+dir*t1;
        // // osg::Vec3 v2 = pos+dir*t2;
        // std::cerr << "    v1: " << v1 << ", v2: " << v2 << std::endl;
    }
    if (t < 0)
    {
        t = -dir * pos;
    }
    if (t >= 0)
    {
        _distance = t * getScale();
        osg::Vec3 isect = pos+dir*t;
        // std::cerr << "valid intersection: t=" << t << ", p=" << isect << ", dist=" << isect.length() << std::endl;
        osg::Matrix rot;
        rot.makeRotate( cylinderDirectionVector, isect);
        interactorXformMat_o = rot * getMatrix();
        
        // restrict rotation to specific axis (therefor we use euler: h=zAxis, p=xAxis, r=yAxis)
        coCoord euler = interactorXformMat_o;
        coCoord Oldeuler = _oldInterMat_o;
        if(rotationAxis == osg::Z_AXIS)
        {
            euler.hpr[1] = Oldeuler.hpr[1]; 
            euler.hpr[2] = Oldeuler.hpr[2]; 
            //std::cout << "diff: " << euler.hpr[0] - _start_o.hpr[0] << " " << std::endl;
        }
        else if(rotationAxis == osg::Y_AXIS)
        {
            euler.hpr[0] = Oldeuler.hpr[0]; 
            euler.hpr[1] = Oldeuler.hpr[1];  
        }
        else if(rotationAxis == osg::X_AXIS)
        {
            euler.hpr[0] = Oldeuler.hpr[0];
            euler.hpr[2] = Oldeuler.hpr[2];
        }
        
        euler.makeMat(interactorXformMat_o);
    }
    else
    {
        //  std::cerr <<"distance = 0"<<std::endl;
        _distance = 0;
    }   

    return interactorXformMat_o;
}
*/

osg::Matrix coVR3DRotGizmo::calcRotation2D(const osg::Vec3& lp0_o, const osg::Vec3& lp1_o, osg::Vec3 rotationAxis)
{
    osg::Matrix interMatrix;
    osg::Vec3 closestPoint;
    _helperPlane->update(rotationAxis*getMatrix(),osg::Vec3(0,0,0)); //fix Vector here !!!!!!!!!!!!!
    closestDistanceLineCircle(lp0_o, lp1_o, rotationAxis*getMatrix(), closestPoint);
    osg::Vec3 dir1 = closestPoint; // hier muss Nulltpunkt abgezogen werden !! -getMatrix.getTrans();
    std::cout <<"closestPoint: " <<closestPoint <<std::endl;
    osg::Vec3 dir2 = _closestStartPoint_o; // hier muss Nulltpunkt abgezogen werden !! -getMatrix.getTrans();
    dir1.normalize();
    dir2.normalize();
    
    osg::Vec3 refVec= rotationAxis*getMatrix();

    osg::Matrix rotate;
    rotate.makeRotate(osg::DegreesToRadians(vecAngle360(dir1, dir2, -refVec)), rotationAxis *getMatrix());
    interMatrix = _startInterMat_w*rotate;

    return interMatrix;
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

void coVR3DRotGizmo::stopInteraction() 
{
    _helperLine->hide();
    coVR3DGizmoType::stopInteraction();
}

