#include <cover/coVRMSController.h>

#include "GizmoTest.h"



using namespace opencover;

GizmoTest::GizmoTest()
{
    _node = osgDB::readNodeFile("/home/AD.EKUPD.COM/matthias.epple/trailer.3ds");
    _transform = new osg::MatrixTransform;
    _transform->addChild(_node.get());
    _gizmo = new GizmoDrawable;
        _gizmo->setScreenSize(30,40);

    _gizmo->setTransform( _transform.get() );
    _gizmo->setGizmoMode( GizmoDrawable::MOVE_GIZMO );  
    _geode = new osg::Geode;
    _geode->addDrawable( _gizmo.get() );
    _geode->setCullingActive( false );  // allow gizmo to always display
    _geode->getOrCreateStateSet()->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );  // always show at last
    _transform->addChild(_gizmo.get());


    cover->getObjectsRoot()->addChild(_transform.get());
}


COVERPLUGIN(GizmoTest)
