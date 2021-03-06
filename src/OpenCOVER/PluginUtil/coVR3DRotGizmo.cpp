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


    osg::Sphere *mySphere = new osg::Sphere(origin, 0.75);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereDrawable->setColor(_grey);
    _sphereGeode = new osg::Geode();
    _sphereGeode->addDrawable(sphereDrawable);
    _axisTransform->addChild(_sphereGeode.get());

    _xRotCylGroup = circlesFromCylinders(RotationAxis::X, 27, _red, _radius/4);
    _yRotCylGroup = circlesFromCylinders(RotationAxis::Y, 27, _green, _radius/4);
    _zRotCylGroup = circlesFromCylinders(RotationAxis::Z, 27, _blue, _radius/4);

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

/*
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
*/

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

    if(!is2D()) //if 3d input device is available create cylinder which is used for rotation via wrist rotation
    {
        auto flatSlice = new osg::Cylinder(osg::Vec3(0,0,0), _radius - 0.4, 0.1);
        osg::ShapeDrawable* flatSliceDrawable = new osg::ShapeDrawable(flatSlice);
        osg::Vec4 newColor = color;
        newColor = osg::Vec4(color.x(),color.y(), color.z(), 0.4);
        flatSliceDrawable->setColor(newColor);
        osg::Geode* geode1 = new osg::Geode;
        geode1->addDrawable(flatSliceDrawable); 
        geode1->setName("UseWristRotation");
        osg::MatrixTransform *matrixTransform = new osg::MatrixTransform();
        osg::Matrix rot;
        if(axis == RotationAxis::Z)
            rot = osg::Matrix::rotate(osg::DegreesToRadians(0.0),osg::Z_AXIS);
        else if(axis == RotationAxis::Y)
            rot = osg::Matrix::rotate(osg::DegreesToRadians(90.0),osg::X_AXIS);
        else if(axis == RotationAxis::X)
            rot = osg::Matrix::rotate(osg::DegreesToRadians(90.0),osg::Y_AXIS);
        matrixTransform->setMatrix(rot);
        matrixTransform->addChild(geode1);
        parent->addChild(matrixTransform);
    }

    return parent;
}

//check if hitNode is child of specific rotation group and derive corresponding rotation axis
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

bool coVR3DRotGizmo::useWristRotation() const
{
    return _hitNode->getName() == "UseWristRotation" ? true : false;
}

void coVR3DRotGizmo::startInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DRotGizmo::startInteraction\n");


    osg::Vec3 lp0_o, lp1_o, pointerDir_o;
    calculatePointerDirection_o(lp0_o, lp1_o, pointerDir_o);
    
    coVR3DGizmoType::startInteraction();

    _rotateZonly = rotateAroundSpecificAxis(_zRotCylGroup.get());
    _rotateYonly = rotateAroundSpecificAxis(_yRotCylGroup.get());
    _rotateXonly = rotateAroundSpecificAxis(_xRotCylGroup.get());
    _wristRotation = useWristRotation();

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
}


void coVR3DRotGizmo::doInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DRotGizmo::rot\n");

    osg::Vec3 lp0_o, lp1_o, pointerDir_o;
    calculatePointerDirection_o(lp0_o, lp1_o, pointerDir_o);

    osg::Matrix newInteractorMatrix;
    if (_rotateZonly)
    {
        if(_wristRotation)
            newInteractorMatrix = calcRotation3D(osg::Z_AXIS);
        else
	        newInteractorMatrix = calcRotation2D(lp0_o, lp1_o, osg::Z_AXIS);
    }
    else if(_rotateYonly)
    {
        if(_wristRotation)
            newInteractorMatrix = calcRotation3D(osg::Y_AXIS);
        else
            newInteractorMatrix = calcRotation2D(lp0_o, lp1_o, osg::Y_AXIS);
    }
    else if(_rotateXonly)
    {
        if(_wristRotation)
            newInteractorMatrix = calcRotation3D(osg::X_AXIS);
        else
            newInteractorMatrix = calcRotation2D(lp0_o, lp1_o, osg::X_AXIS);            
    }
    else if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors)
    {
        // move old mat to hand position, apply rel hand movement and move it back to
       // newInteractorMatrix = _oldInterMat_o * transToHand_o * relHandMoveMat_o * revTransToHand_o;
    }
    else
    {
        // if sphere in the center is selected apply hand rotation to the gizmo
        osg::Matrix invStartHandMat;
        invStartHandMat.invert(_startHandMat_w);
        osg::Matrix currentHandMat = getPointerMat();
        osg::Matrix diffMat = invStartHandMat * currentHandMat;
        newInteractorMatrix = _startInterMat_w * diffMat;
        newInteractorMatrix.setTrans(_startInterMat_w.getTrans());

        if (coVRNavigationManager::instance()->isSnapping())
        {
            if (coVRNavigationManager::instance()->isDegreeSnapping())
                snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &newInteractorMatrix);
            else
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
    float angle = vecAngle360(dir1, dir2, -normal);
    if (coVRNavigationManager::instance()->isSnapping())
    {
        snapTo45(angle);
        totalRotation.makeRotate(osg::DegreesToRadians(angle), normal);
    }
    else
        totalRotation.makeRotate(osg::DegreesToRadians(angle), normal);

    osg::Matrix startRotation = _startInterMat_w;
    startRotation.setTrans(osg::Vec3(0,0,0));
    
    return startRotation * totalRotation * osg::Matrix::translate(_startInterMat_w.getTrans());
}

osg::Matrix coVR3DRotGizmo::calcRotation3D(osg::Vec3 rotationAxis)
{
    
    osg::Matrix interMatrix;
    coCoord startEuler = _startHandMat_w;
    coCoord currentEuler = getPointerMat();

    float angle = currentEuler.hpr[2] - startEuler.hpr[2];
    
    osg::Matrix startRotationOnly = osg::Matrix::rotate(_startInterMat_w.getRotate());

    // Use sign of the Spatprodukt (between pointer direction and normal) to calculate the direction of the rotation
	osg::Vec3 lp0,lp1,pointerDir;
	osg::Vec3 normal = rotationAxis * startRotationOnly;
	calculatePointerDirection_o(lp0,lp1,pointerDir);
	float spat = normal.operator*(pointerDir);
    int sign;
	if(spat>=0.0)
		sign=1;
	else
		sign=-1;
    
    osg::Matrix rotate;
    if (coVRNavigationManager::instance()->isSnapping())
    {
        snapTo45(angle);
        rotate.makeRotate(sign * osg::DegreesToRadians(angle), normal);
    }
    else
        rotate.makeRotate(sign * osg::DegreesToRadians(angle), normal);

    interMatrix = startRotationOnly*rotate * osg::Matrix::translate(_startInterMat_w.getTrans());  
		
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

void coVR3DRotGizmo::snapTo45(float& angle)const
{
    int factor = floor(std::ceil(angle)/ 45);
    angle = (float)factor * 45;
}

void coVR3DRotGizmo::stopInteraction() 
{
    _helperLine->hide();
    coVR3DGizmoType::stopInteraction();
}

