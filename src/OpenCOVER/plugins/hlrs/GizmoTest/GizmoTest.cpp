#include <cover/coVRMSController.h>
#include<osg/ShapeDrawable>

#include "GizmoTest.h"
#include <osgSim/SphereSegment>

#include <cover/VRSceneGraph.h>
#include <osg/Material>



using namespace opencover;

void GizmoTest::preFrame()
{
    _transgizmo->preFrame();
    _rotgizmo->preFrame();
    _scalegizmo->preFrame();
    _transRotInteractor->preFrame();
    _gizmo->preFrame();
}

GizmoTest::GizmoTest()
{
   
    osg::Box *unitCube1 = new osg::Box(osg::Vec3(0, 0, 0), 10.0f);
    osg::ShapeDrawable *unitCubeDrawable1 = new osg::ShapeDrawable(unitCube1);


    _cube1 = new osg::Geode();
    _cube1->setName("cube1");

    _cube1->addDrawable(unitCubeDrawable1);


    _scene = new osg::MatrixTransform;
    _scene->setName("Scene");
    _scene->setMatrix(osg::Matrix::translate(osg::Vec3(0,20,0)));
    _scene->addChild(_cube1.get());

   
    osg::Matrix matrix; 
    matrix.makeRotate(osg::DegreesToRadians(-130.0), osg::Z_AXIS);
    //osg::Matrix::translate(osg::Vec3(40,0,0));
    float _interSize = cover->getSceneSize() / 50 ;
    _transgizmo = new coVR3DTransGizmo(matrix*osg::Matrix::translate(osg::Vec3(40,0,0)), _interSize, vrui::coInteraction::ButtonA, "hand", "CamInteractor", vrui::coInteraction::Medium);
    _transgizmo->setName("Gizmo");
    _transgizmo->enableIntersection();
    _transgizmo->show();

    osg::Matrix matrix2 = osg::Matrix::translate(osg::Vec3(0,0,0)); 
    //matrix2.makeRotate(osg::DegreesToRadians(30.0), osg::X_AXIS);
    _rotgizmo = new coVR3DRotGizmo(matrix2, _interSize, vrui::coInteraction::ButtonA, "hand", "CamInteractor", vrui::coInteraction::Medium);
    _rotgizmo->setName("RotGizmo");
    _rotgizmo->enableIntersection();
    _rotgizmo->show();

    osg::Matrix matrix3 = osg::Matrix::translate(osg::Vec3(-40,0,0)); 
    _scalegizmo = new coVR3DScaleGizmo(matrix3, _interSize, vrui::coInteraction::ButtonA, "hand", "CamInteractor", vrui::coInteraction::Medium);
    _scalegizmo->setName("ScaleGizmo");
    _scalegizmo->enableIntersection();
    _scalegizmo->show();

    osg::Matrix matrix4 = osg::Matrix::translate(osg::Vec3(0,0,40)); 
    _transRotInteractor = new coVR3DTransRotInteractor(matrix4, _interSize, vrui::coInteraction::ButtonA, "hand", "CamInteractor", vrui::coInteraction::Medium);
    _transRotInteractor->setName("TransRot");
    _transRotInteractor->enableIntersection();
    _transRotInteractor->show();

    osg::Matrix matrix5 = osg::Matrix::translate(osg::Vec3(0,0,-40)); 
    _gizmo = new coVR3DGizmo(coVR3DGizmo::GIZMO_TYPE::ROTATE, matrix5, _interSize, vrui::coInteraction::ButtonA, "hand", "CamInteractor", vrui::coInteraction::Medium);
    





    // osgSim::SphereSegment *circleGeode = new osgSim::SphereSegment(osg::Vec3(0,0,0),1.0,0.0,M_PI/10,0,M_PI,32);
    // circleGeode->setDrawMask(osgSim::SphereSegment::DrawMask(osgSim::SphereSegment::SURFACE));
    // circleGeode->setAllColors(osg::Vec4(1,0,0,1));
    // circleGeode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    // cover->getObjectsRoot()->addChild(circleGeode);

    //cover->getObjectsRoot()->addChild(_root.get());
    cover->getObjectsRoot()->addChild(_scene.get());



    //_circle = circles(0,32);
    //cover->getObjectsRoot()->addChild(_circle.get());
}

COVERPLUGIN(GizmoTest)
