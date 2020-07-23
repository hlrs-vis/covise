#include <cover/coVRMSController.h>
#include<osg/ShapeDrawable>

#include "GizmoTest.h"



using namespace opencover;

GizmoTest::GizmoTest()
{
    //_node = osgDB::readNodeFile("/home/AD.EKUPD.COM/matthias.epple/trailer.3ds");
    
    osg::Box *unitCube1 = new osg::Box(osg::Vec3(0, 0, 0), 20.0f);
    osg::ShapeDrawable *unitCubeDrawable1 = new osg::ShapeDrawable(unitCube1);


    _cube1 = new osg::Geode();
    _cube2 = new osg::Geode();
    _cube1->setName("cube1");
    _cube2->setName("cube2");

    _cube1->addDrawable(unitCubeDrawable1);
    _cube2->addDrawable(unitCubeDrawable1);


    _scene = new osg::MatrixTransform;
    _scene->setName("Scene");
    //_scene->setMatrix(osg::Matrix::translate(osg::Vec3(200,200,200)));
    _scene->addChild(_cube1.get());
    //_scene->addChild(_node.get());

    _gizmo = new GizmoDrawable;
    _gizmo->setTransform( _scene.get() );
    _gizmo->setGizmoMode( GizmoDrawable::MOVE_GIZMO );  

    _gizmoGeode = new osg::Geode;
    _gizmoGeode->setName("Gizmo");
    _gizmoGeode->addDrawable( _gizmo.get() );
    _gizmoGeode->setCullingActive( false );  // allow gizmo to always display
    _gizmoGeode->getOrCreateStateSet()->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );  // always show at last
    
    _root = new osg::MatrixTransform;
    _root->addChild(_scene.get());
    _root->addChild(_gizmoGeode.get());
    //_root->addChild(_cube2.get());


    cover->getObjectsRoot()->addChild(_root.get());
}


COVERPLUGIN(GizmoTest)
