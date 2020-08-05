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
    :coVRIntersectionInteractor(s, type, iconName, interactorName, priority)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR3DRotGizmo(%s)\n", interactorName);
    }

    createGeometry();

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

    _rotateXaxisGeode = circles(1,32,red);
    _rotateYaxisGeode = circles(2,32,green);
    _rotateZaxisGeode = circles(3,32,blue);

    _xAxisTransform = new osg::MatrixTransform();
    _yAxisTransform = new osg::MatrixTransform();
    _zAxisTransform = new osg::MatrixTransform();

    _xAxisTransform->addChild(_rotateXaxisGeode.get());
    _yAxisTransform->addChild(_rotateYaxisGeode.get());
    _zAxisTransform->addChild(_rotateZaxisGeode.get());


    _axisTransform->addChild(_xAxisTransform);
    _axisTransform->addChild(_yAxisTransform);
    _axisTransform->addChild(_zAxisTransform);

    //temporary cylinders:
    osg::ShapeDrawable *xCylDrawable,*yCylDrawable,*zCylDrawable;
    auto cyl = new osg::Cylinder(origin, 0.15, _radius*2);
    xCylDrawable = new osg::ShapeDrawable(cyl, hint);
    yCylDrawable = new osg::ShapeDrawable(cyl, hint);
    zCylDrawable = new osg::ShapeDrawable(cyl, hint);
    
    xCylDrawable->setColor(red);
    yCylDrawable->setColor(green);
    zCylDrawable->setColor(blue);
        
     
    _xRotCylTransform = new osg::MatrixTransform();
    //_xRotCylTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(0.0), 1, 0, 0));
    _yRotCylTransform = new osg::MatrixTransform();
    _yRotCylTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(90.0), 0, 1, 0));
    _zRotCylTransform = new osg::MatrixTransform();
    _zRotCylTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(90.0), 1, 0, 0));

       
    _xRotCylGeode = new osg::Geode;
    _xRotCylGeode->addDrawable(xCylDrawable);
    _xRotCylTransform->addChild(_xRotCylGeode); 
    _yRotCylGeode = new osg::Geode;
    _yRotCylGeode->addDrawable(yCylDrawable);
    _yRotCylTransform->addChild(_yRotCylGeode);  
    _zRotCylGeode = new osg::Geode;
    _zRotCylGeode->addDrawable(zCylDrawable);
    _zRotCylTransform->addChild(_zRotCylGeode);
        
    _axisTransform->addChild(_xRotCylTransform);
    _axisTransform->addChild(_yRotCylTransform);
    _axisTransform->addChild(_zRotCylTransform);

    
}

osg::Vec3Array* coVR3DRotGizmo::circleVerts(int plane, int approx)
{
    const double angle( osg::PI * 2. / (double) approx );
    osg::Vec3Array* v = new osg::Vec3Array;
    int idx;
    for( idx=0; idx<approx; idx++)
    {
        double cosAngle = cos(idx*angle);
        double sinAngle = sin(idx*angle);
        double x(0.), y(0.), z(0.);
        switch (plane) {
            case 1 : // X
                y = cosAngle*_radius;
                z = sinAngle*_radius;
                break;
            case 2 : // Y
                x = cosAngle*_radius;
                z = sinAngle*_radius;
                break;
            case 3: // Z
                x = cosAngle*_radius;
                y = sinAngle*_radius;
                break;
        }
        v->push_back( osg::Vec3( x, y, z ) );
    }
    return v;
}

osg::Geode* coVR3DRotGizmo:: circles( int plane, int approx, osg::Vec4 color )
{
osg::Geode* geode = new osg::Geode;
osg::LineWidth* lw = new osg::LineWidth( 4. );
geode->getOrCreateStateSet()->setAttributeAndModes( lw,
osg::StateAttribute::ON );


osg::Geometry* geom = new osg::Geometry;
osg::Vec3Array* v = circleVerts( plane, approx );
geom->setVertexArray( v );

osg::Vec4Array* c = new osg::Vec4Array;
c->push_back( color );
geom->setColorArray( c );
geom->setColorBinding( osg::Geometry::BIND_OVERALL );
geom->addPrimitiveSet( new osg::DrawArrays( GL_LINE_LOOP, 0, approx ) );

geode->addDrawable( geom );
return geode;
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

    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix o_to_w = cover->getBaseMat();

    osg::Matrix hm = getPointerMat(); // hand matrix weltcoord
    osg::Matrix hm_o = hm * w_to_o;   // hand matrix objekt coord
    _oldHandMat = hm;
    _invOldHandMat_o.invert(hm_o); // store the inv hand matrix

    osg::Matrix interMat = _interMat_o * o_to_w;

    _oldInteractorXformMat_o = _interMat_o;

    osg::Vec3 interPos = getMatrix().getTrans();
    // get diff between intersection point and sphere center
    _diff = interPos - _hitPos;
    _distance = (_hitPos - hm_o.getTrans()).length();

    _rotateZonly = _hitNode == _zRotCylGeode;
    _rotateYonly = _hitNode == _yRotCylGeode;
    _rotateXonly = _hitNode == _xRotCylGeode;



    // if (!_rotateOnly && !_translateOnly)
    // {
    //     _translateOnly = is2D();
    // }

    coVRIntersectionInteractor::startInteraction();

}

void coVR3DRotGizmo::doInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DRotGizmo::rot\n");

    osg::Vec3 origin(0, 0, 0);


    osg::Matrix currHandMat = getPointerMat();
    // forbid translation in y-direction if traverseInteractors is on --> why do we need this ???? ###############################
    // if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRConfig::instance()->useWiiNavigationVisenso())
    // {
    //     osg::Vec3 trans = currHandMat.getTrans();
    //     trans[1] = _oldHandMat.getTrans()[1];
    //     currHandMat.setTrans(trans);
    // }

    osg::Matrix o_to_w = cover->getBaseMat();
    // get hand mat in object coords
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;

    // translate from interactor to hand and back
    osg::Matrix transToHand_o, revTransToHand_o;

    transToHand_o.makeTranslate(currHandMat_o.getTrans() - _oldInteractorXformMat_o.getTrans());
    revTransToHand_o.makeTranslate(_oldInteractorXformMat_o.getTrans() - currHandMat_o.getTrans());

    osg::Matrix relHandMoveMat_o = _invOldHandMat_o * currHandMat_o;

    osg::Matrix interactorXformMat_o = _oldInteractorXformMat_o;
    if (_rotateZonly)
        interactorXformMat_o = calcRotation(osg::Z_AXIS, osg::Vec3(0, -1, 0));
    else if(_rotateYonly)
        interactorXformMat_o = calcRotation(osg::Y_AXIS, osg::Vec3(-1, 0, 0));
    else if(_rotateXonly)
        interactorXformMat_o = calcRotation(osg::X_AXIS, osg::Vec3(0, 0, 1));

    else if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors)
    {
        // move old mat to hand position, apply rel hand movement and move it back to
        interactorXformMat_o = _oldInteractorXformMat_o * transToHand_o * relHandMoveMat_o * revTransToHand_o;
    }
    else
    {
        // apply rel hand movement
        interactorXformMat_o = _oldInteractorXformMat_o * relHandMoveMat_o;
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

osg::Matrix coVR3DRotGizmo::calcRotation(osg::Vec3 rotationAxis, osg::Vec3 cylinderDirectionVector)
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

void coVR3DRotGizmo::updateTransform(osg::Matrix m)
{
    if (cover->debugLevel(5))
    //fprintf(stderr, "coVR3DTransGizmo:setMatrix\n");
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
