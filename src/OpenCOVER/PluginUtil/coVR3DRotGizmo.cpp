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
#include <vrbclient/SharedState.h>

using namespace opencover;

coVR3DRotGizmo::coVR3DRotGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium)
    :coVRIntersectionInteractor(s, type, iconName, interactorName, priority, true)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR3DRotGizmo(%s)\n", interactorName);
    }

    createGeometry();

    _plane.reset(new opencover::coPlane(osg::Vec3(0.0, 0.0, 1.0), osg::Vec3(0.0, 0.0, 0.0))); //add getMatrix
    _line.reset(new opencover::coLine(osg::Vec3(0.0, 0.0, 1.0)*getMatrix(), osg::Vec3(0.0, 0.0, 0.0)*getMatrix()));



    coVR3DRotGizmo::updateTransform(m);

    
}

coVR3DRotGizmo::~coVR3DRotGizmo()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~coVR3DRotGizmo\n");
}

void coVR3DRotGizmo::createGeometry()
{
    osg::ShapeDrawable *sphereDrawable;
    osg::Vec4 red(0.5, 0.2, 0.2, 1.0), green(0.2, 0.5, 0.2, 1.0), blue(0.2, 0.2, 0.5, 1.0), color(0.5, 0.5, 0.5, 1);
    osg::Vec3 origin(0, 0, 0);

    _axisTransform = new osg::MatrixTransform();
    _axisTransform->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    geometryNode = _axisTransform.get();
    scaleTransform->addChild(geometryNode.get());


    osg::Sphere *mySphere = new osg::Sphere(origin, 0.5);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereDrawable->setColor(color);
    _sphereGeode = new osg::Geode();
    _sphereGeode->addDrawable(sphereDrawable);
    _axisTransform->addChild(_sphereGeode.get());

    _xRotCylGroup = circlesFromCylinders(RotationAxis::X, 24, red, _radius/4);
    _yRotCylGroup = circlesFromCylinders(RotationAxis::Y, 24, green, _radius/4);
    _zRotCylGroup = circlesFromCylinders(RotationAxis::Z, 24, blue, _radius/4);

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

void coVR3DRotGizmo::updateSharedState()
{
    // if (auto st = static_cast<SharedMatrix *>(m_sharedState.get()))
    // {
        // *st = _oldInteractorXformMat_o;//myPosition
    // }
}

void coVR3DRotGizmo::startInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DRotGizmo::startInteraction\n");

    osg::Vec3 origin(0, 0, 0);
    osg::Vec3 yaxis(0, 1, 0);
    osg::Matrix o_to_w = cover->getBaseMat();
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat = getPointerMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;
    osg::Matrix interactorXformMat_o = _oldInteractorXformMat_o;
    startAngle_o = interactorXformMat_o;
    _startMatrix = getMatrix();//interactorXformMat_o;
    _oldInteractorXformMat_o = _startMatrix;


    // pointer direction in world coordinates
    osg::Vec3 lp0 = origin * currHandMat;
    osg::Vec3 lp1 = yaxis *currHandMat;
    // pointer direction in object coordinates
    osg::Vec3 lp0_o = lp0 * w_to_o;
    osg::Vec3 lp1_o = lp1 * w_to_o;
    
   coCoord euler = getMatrix();
   //std::cout << "euler startInteraction"<< "z: " <<euler.hpr[0] << " x:" <<euler.hpr[1] << "y: " <<euler.hpr[2] <<std::endl; 

/*
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix o_to_w = cover->getBaseMat();

    osg::Matrix hm = getPointerMat(); // hand matrix weltcoord
    osg::Matrix hm_o = hm * w_to_o;   // hand matrix objekt coord
    _oldHandMat = hm;
    _invOldHandMat_o.invert(hm_o); // store the inv hand matrix

    osg::Matrix interMat = _interMat_o * o_to_w;

    _oldInteractorXformMat_o = _interMat_o;
    _start_o = hm;

 */   osg::Vec3 interPos = getMatrix().getTrans();
    // get diff between intersection point and sphere center
    _diff = interPos - _hitPos;
    _distance = (_hitPos - currHandMat_o.getTrans()).length();

    _rotateZonly = rotateAroundSpecificAxis(_zRotCylGroup.get());
    _rotateYonly = rotateAroundSpecificAxis(_yRotCylGroup.get());
    _rotateXonly = rotateAroundSpecificAxis(_xRotCylGroup.get());

    if(_rotateZonly)
    {
        _plane->update(osg::Z_AXIS * getMatrix(),osg::Vec3(0,0,0)*getMatrix());
        _startPos = calcPlaneLineIntersection(lp0_o, lp1_o, osg::Z_AXIS * getMatrix());
        closestDistanceLineCircle(lp0_o, lp1_o, osg::Z_AXIS * getMatrix(), _result_o);

        _line->setColor(osg::Vec4(0.2, 0.2, 0.5, 1.0));
        _line->update(osg::Vec3(0,0,-_radius*3*getScale())*getMatrix(),osg::Vec3(0,0,_radius*3*getScale())*getMatrix());
        _line->show();

        

    }
    else if(_rotateYonly)
    {
        _plane->update(osg::Y_AXIS * getMatrix(),osg::Vec3(0,0,0));
        _startPos = calcPlaneLineIntersection(lp0_o, lp1_o, osg::Y_AXIS* getMatrix());
        closestDistanceLineCircle(lp0_o, lp1_o, osg::Y_AXIS* getMatrix(), _result_o);

        _line->setColor(osg::Vec4(0.2, 0.5, 0.2, 1.0));
        _line->update(osg::Vec3(0,-_radius*3*getScale(),0)*getMatrix(),osg::Vec3(0,_radius*3*getScale(),0)*getMatrix());
        _line->show();
    }
    else if(_rotateXonly)
    {
        _plane->update(osg::X_AXIS * getMatrix(),osg::Vec3(0,0,0));
        _startPos = calcPlaneLineIntersection(lp0_o, lp1_o, osg::X_AXIS* getMatrix());
        closestDistanceLineCircle(lp0_o, lp1_o, osg::X_AXIS* getMatrix(), _result_o);

        _line->setColor(osg::Vec4(0.5, 0.2, 0.2, 1.0));
        _line->update(osg::Vec3(-_radius*3*getScale(),0,0)*getMatrix(),osg::Vec3(_radius*3*getScale(),0,0)*getMatrix());
        _line->show();
    }
   

    // if (!_rotateOnly && !_translateOnly)
    // {
    //     _translateOnly = is2D();
    // }

    coVRIntersectionInteractor::startInteraction();

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
    // osg::Matrix interactorXformMat_o = _oldInteractorXformMat_o;


    // pointer direction in world coordinates
    osg::Vec3 lp0 = origin * currHandMat;
    osg::Vec3 lp1 = yaxis *currHandMat;
    // pointer direction in object coordinates
    osg::Vec3 lp0_o = lp0 * w_to_o;
    osg::Vec3 lp1_o = lp1 * w_to_o;
    

    // forbid translation in y-direction if traverseInteractors is on --> why do we need this ???? ###############################
    // if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRConfig::instance()->useWiiNavigationVisenso())
    // {
    //     osg::Vec3 trans = currHandMat.getTrans();
    //     trans[1] = _oldHandMat.getTrans()[1];
    //     currHandMat.setTrans(trans);
    // }


    // translate from interactor to hand and back
    osg::Matrix transToHand_o, revTransToHand_o;

    transToHand_o.makeTranslate(currHandMat_o.getTrans() - _oldInteractorXformMat_o.getTrans());
    revTransToHand_o.makeTranslate(_oldInteractorXformMat_o.getTrans() - currHandMat_o.getTrans());

    osg::Matrix relHandMoveMat_o = _invOldHandMat_o * currHandMat_o;
    //std::cout << "rel Hand Movement:" <<relHandMoveMat_o.getTrans() <<"..."<< std::endl;
    osg::Matrix interactorXformMat_o = _oldInteractorXformMat_o;
    static double sumAngle{0.0};
    if (_rotateZonly)
    {
        //if(is2D())
          //  interactorXformMat_o = calcRotation2D(osg::Z_AXIS, osg::Vec3(0, -1, 0));
        //else 
        //    interactorXformMat_o = calcRotation3D(osg::Z_AXIS);
        _plane->update(osg::Z_AXIS*getMatrix(),osg::Vec3(0,0,0)); //ffix vec here !!!

        osg::Vec3 result;
        closestDistanceLineCircle(lp0_o, lp1_o, osg::Z_AXIS*getMatrix(), result);
        osg::Vec3 dir1 = result;// - getMatrix().getTrans();
        osg::Vec3 dir2 = _result_o;// - getMatrix().getTrans();
        dir1.normalize();
        dir2.normalize();
        double dotProduct = dir1 * dir2;
        if(dotProduct > 1.0) //catch calculation imprecision where dotproduct is smaller / bigger +-1 (acos is only defined for the range between -1 and 1 )
            dotProduct = 1.0;
        else if(dotProduct < -1.0)
            dotProduct = -1.0;
        double angle_rad = std::acos(dotProduct);
        double angle_deg = osg::RadiansToDegrees(angle_rad);
        // std::cout <<"dir1 * dir2 (dot)"<<dir1 * dir2 <<std::endl;
        // std::cout << "dir1: " << dir1 << " dir2: " << dir2<<std::endl;
        // std::cout << "angle deg: " << angle_deg << " angle rad: " << angle_rad<<std::endl;
        //_result_o = result;
        if(std::isnan(angle_deg))
            angle_deg = 0.0f;
        if(std::isnan(angle_deg))
            angle_rad = 0.0f;

        osg::Vec3 x = dir1^dir2;
        int sign;
        double c;
        osg::Vec3 refVec=osg::Vec3(0,0,1) * getMatrix();
        if(x*refVec > 0)
            sign = 1;
        else if (x*refVec <0)
            sign = -1;
        //x.normalize();
        c = sign * x.length();
        double a = osg::RadiansToDegrees( atan2(c,dir1*dir2) );
        // std::cout << " a" << a << "..."  <<std::endl;


        // sumAngle += angle_deg;
        // std::cout <<"sum angle: " << sumAngle <<std::endl;
        // osg::Matrix rotate;
        // rotate.makeRotate(angle_deg, osg::Z_AXIS);
        // interactorXformMat_o =    getMatrix() * rotate;
        //static bool bigger{false};
        // if(angle_deg > 175.0)
            // bigger = true;
 
        // if(bigger)
            // angle_deg = 175 + 175 - angle_deg;

        coCoord euler = interactorXformMat_o;
        //euler.hpr[0] =  startAngle_o.hpr[0] + angle_deg;
        //osg::Vec3 refVec= osg::Vec3(0,0,1) *getMatrix();
        euler.hpr[0] =  startAngle_o.hpr[0] + vecAngle360(dir1, dir2, -refVec);

       
        osg::Matrix rotate;
        double angle = osg::DegreesToRadians(vecAngle360(dir1, dir2, -refVec));
        //std::cout <<"angle: " << osg::RadiansToDegrees( angle ) <<std::endl;
        rotate.makeRotate(angle, osg::Z_AXIS *getMatrix());
        interactorXformMat_o = _startMatrix*rotate;

        

 
        //euler.makeMat(interactorXformMat_o);
        // interactorXformMat_o = interactorXformMat_o * osg::Matrix::rotate(angle_rad, osg::Z_AXIS);
        
    }
    else if(_rotateYonly)
    {
        if(is2D())
        {   
            _plane->update(osg::Y_AXIS * getMatrix(), osg::Vec3(0,0,0));
            osg::Vec3 result;
            closestDistanceLineCircle(lp0_o, lp1_o, osg::Y_AXIS*getMatrix(), result);
            osg::Vec3 dir1 = result;// - getMatrix().getTrans();
            osg::Vec3 dir2 = _result_o;// - getMatrix().getTrans();
            dir1.normalize();
            dir2.normalize();

            coCoord euler = interactorXformMat_o;
            osg::Vec3 refVec= osg::Vec3(0,1,0)*getMatrix();
            // euler.hpr[2] =  startAngle_o.hpr[2] + vecAngle360(dir1, dir2, -refVec);
            // euler.makeMat(interactorXformMat_o);

            osg::Matrix rotate;
            rotate.makeRotate(osg::DegreesToRadians(vecAngle360(dir1, dir2, -refVec)), osg::Y_AXIS *getMatrix());
            interactorXformMat_o = _startMatrix*rotate;

            //interactorXformMat_o = calcRotation2D(osg::Y_AXIS, osg::Vec3(-1, 0, 0));

        }
        else
            interactorXformMat_o = calcRotation3D(osg::Y_AXIS);

    }
    else if(_rotateXonly)
    {
        if(is2D())
        {
            _plane->update(osg::X_AXIS*getMatrix(),osg::Vec3(0,0,0)); //ffix vec here !!!
            osg::Vec3 result;
            closestDistanceLineCircle(lp0_o, lp1_o, osg::X_AXIS*getMatrix(), result);
            osg::Vec3 dir1 = result;// - getMatrix().getTrans();
            osg::Vec3 dir2 = _result_o;// - getMatrix().getTrans();
            dir1.normalize();
            dir2.normalize();

            coCoord euler = interactorXformMat_o;
            osg::Vec3 refVec = osg::Vec3(1,0,0)*getMatrix();
            // euler.hpr[1] =  startAngle_o.hpr[1] + vecAngle360(dir1, dir2, refVec);
            // euler.makeMat(interactorXformMat_o);

            osg::Matrix rotate;
            rotate.makeRotate(osg::DegreesToRadians(vecAngle360(dir1, dir2, -refVec)), osg::X_AXIS *getMatrix());
            interactorXformMat_o = _startMatrix*rotate;
            //interactorXformMat_o = calcRotation2D(osg::X_AXIS, osg::Vec3(0, 0, 1));
        }
        else
            interactorXformMat_o = calcRotation3D(osg::X_AXIS);
    }
    else if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors)
    {
        // move old mat to hand position, apply rel hand movement and move it back to
        interactorXformMat_o = _oldInteractorXformMat_o * transToHand_o * relHandMoveMat_o * revTransToHand_o;
    }
    else
    {
        //if(!is2D())
            interactorXformMat_o = _oldInteractorXformMat_o * relHandMoveMat_o; // apply rel hand movement

    }

    // save old transformation
    _oldInteractorXformMat_o = interactorXformMat_o;

    _oldHandMat = currHandMat; // save current hand for rotation start
    _invOldHandMat_o.invert(currHandMat_o);

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


    // and now we apply it
     updateTransform(interactorXformMat_o);

}

osg::Vec3 coVR3DRotGizmo::calcPlaneLineIntersection(const osg::Vec3& lp0, const osg::Vec3& lp1, osg::Vec3 fixAxis) const
{
    osg::Vec3 isectPoint, newPos;
    _plane->update(fixAxis, osg::Vec3(0,0,0));//--> das hier fixen! // FIXME: Orientierung multiplizieren mit Axen !!!!?
    bool intersect = _plane->getLineIntersectionPoint( lp0, lp1, isectPoint);
    newPos  = isectPoint + _diff;

    if(fixAxis == osg::X_AXIS)
        newPos.x() = _oldInteractorXformMat_o.getTrans().x();
    else if(fixAxis == osg::Y_AXIS)
        newPos.y() = _oldInteractorXformMat_o.getTrans().y();
    else if(fixAxis == osg::Z_AXIS)
        newPos.z() = _oldInteractorXformMat_o.getTrans().z();

    return newPos; //FIXME what happens if lines are parallel ? 
}


float coVR3DRotGizmo::closestDistanceLineCircle(const osg::Vec3& lp0, const osg::Vec3& lp1,const osg::Vec3 rotationAxis, osg::Vec3 &closestPoint) const
{
    osg::Vec3 isectPoint;
    _plane->update(rotationAxis, getMatrix().getTrans());
    bool intersect = _plane->getLineIntersectionPoint( lp0, lp1, isectPoint);
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

osg::Geode* coVR3DRotGizmo::createLine(osg::Vec3 point1, osg::Vec3 point2, osg::Vec4 color)
{
    osg::Geode *geode = new osg::Geode();
    geode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    osg::Geometry* lineGeom = new osg::Geometry();
    osg::Vec3Array* vertices = new osg::Vec3Array(2);
    // osg::Vec3 point1 = osg::Vec3(0,0,-10) *getMatrix() ;
    // osg::Vec3 point2 = osg::Vec3(0,0,10) *getMatrix();
    

    (*vertices)[0].set(point1);
    (*vertices)[1].set(point2);

    lineGeom->setVertexArray(vertices);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(color);
    lineGeom->setColorArray(colors);
    lineGeom->setColorBinding(osg::Geometry::BIND_OVERALL);

    lineGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,2));

    geode->addDrawable(lineGeom);

    return geode;
}


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
        coCoord Oldeuler = _oldInteractorXformMat_o;
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


osg::Matrix coVR3DRotGizmo::calcRotation3D(osg::Vec3 rotationAxis)
{
    /*
    coCoord startEuler, currentEuler, diffEuler; // = startRotMatrix

    std::cout << "euler Start: z " << startEuler.hpr[0] <<" x " << startEuler.hpr[1] <<" y " << startEuler.hpr[2] << " ... "<<std::endl;

    // restrict rotation to specific axis (therefor we use euler: h=zAxis, p=xAxis, r=yAxis)
    coCoord euler = interactorXformMat_o;
    coCoord Oldeuler = _oldInteractorXformMat_o;
    if(rotationAxis == osg::Z_AXIS)
    {
        euler.hpr[1] = Oldeuler.hpr[1]; 
        euler.hpr[2] = Oldeuler.hpr[2]; 
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

    diffEuler.hpr = currentEuler.hpr - startEuler.hpr; 
    std::cout << "current Euler: z " << currentEuler.hpr[0] <<" x " << currentEuler.hpr[1] <<" y " << currentEuler.hpr[2] << " ... "<<std::endl;

    std::cout << "Diff Euler: z " << diffEuler.hpr[0] << " x " << diffEuler.hpr[1] <<" y " << diffEuler.hpr[2] << " ... "<<std::endl;

    return _oldInteractorXformMat_o;
*/
}


void coVR3DRotGizmo::updateTransform(osg::Matrix m)
{
    if (cover->debugLevel(5))
    //fprintf(stderr, "coVR3DTransGizmo:setMatrix\n");
    _interMat_o = m;
    moveTransform->setMatrix(m);
    coCoord eulerMove = moveTransform-> getMatrix();

    if (m_sharedState)
    {
    if (auto st = static_cast<SharedMatrix *>(m_sharedState.get()))
    {
      *st = m;
    }
    }

    // coCoord euler1 =m;
    // std::cout << "euler constructor first"<< "z: " <<euler1.hpr[0] << " x:" <<euler1.hpr[1] << "y: " <<euler1.hpr[2] <<std::endl;
    // coCoord euler = getMatrix();
    // std::cout << "euler constructor late"<< "z: " <<euler.hpr[0] << " x:" <<euler.hpr[1] << "y: " <<euler.hpr[2] <<std::endl; 
    // coCoord interMat = _interMat_o;
    // std::cout << "interMat_o"<< "z: " <<interMat.hpr[0] << " x:" <<interMat.hpr[1] << "y: " <<interMat.hpr[2] <<std::endl; 
// 
    // std::cout << "euler Move"<< "z: " <<eulerMove.hpr[0] << " x:" <<eulerMove.hpr[1] << "y: " <<eulerMove.hpr[2] <<std::endl; 
}

void coVR3DRotGizmo::setShared(bool shared)
{
    // if (shared)
    // {
    //     if (!m_sharedState)
    //     {
    //         m_sharedState.reset(new SharedMatrix("interactor." + std::string(_interactorName), _oldInteractorXformMat_o));//myPosition
    //         m_sharedState->setUpdateFunction([this]() {
    //             m_isInitializedThroughSharedState = true;
    //             osg::Matrix interactorXformMat_o = *static_cast<SharedMatrix *>(m_sharedState.get());
    //             if (cover->restrictOn())
    //             {
    //                 // restrict to visible scene
    //                 osg::Vec3 pos_o, restrictedPos_o;
    //                 pos_o = interactorXformMat_o.getTrans();
    //                 restrictedPos_o = restrictToVisibleScene(pos_o);
    //                 interactorXformMat_o.setTrans(restrictedPos_o);
    //             }

    //             if (coVRNavigationManager::instance()->isSnapping())
    //             {
    //                 if (coVRNavigationManager::instance()->isDegreeSnapping())
    //                 {
    //                     // snap orientation
    //                     snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &interactorXformMat_o);
    //                 }
    //                 else
    //                 {
    //                     // snap orientation to 45 degree
    //                     snapTo45Degrees(&interactorXformMat_o);
    //                 }
    //             }
    //             updateTransform(interactorXformMat_o);
    //         });
    //     }
    // }
    // else
    // {
    //     m_sharedState.reset(nullptr);
    // }
}
void coVR3DRotGizmo::preFrame()
{
    coVRIntersectionInteractor::preFrame();
}

void coVR3DRotGizmo::stopInteraction() 
{
    
    _line->hide();
    coVRIntersectionInteractor::stopInteraction();
}

