/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Brickwall_PLUGIN_H
#define _Brickwall_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Hello OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>

#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <osgViewer/Viewer>
#include <osgDB/ReadFile>
#include <osgDB/FileNameUtils>
#include <osgDB/FileUtils>
#include <osgwTools/AbsoluteModelTransform.h>
#include <osg/ShapeDrawable>
#include <osg/Geode>

#include <btBulletDynamicsCommon.h>

#include <osgbCollision/GLDebugDrawer.h>
#include <osgbCollision/Version.h>
#include <osgbDynamics/MotionState.h>
#include <osgbDynamics/PhysicsState.h>
#include <osgbCollision/CollisionShapes.h>
#include <osgbCollision/RefBulletObject.h>
#include <osgbDynamics/RigidBody.h>
#include <osgbCollision/Utils.h>

#include <osgbInteraction/SaveRestoreHandler.h>
#include <osgbInteraction/DragHandler.h>


#include <osg/io_utils>
#include <iostream>
#include <sstream>
#include <osg/Material>

#include <osgViewer/ViewerEventHandlers>

#include <osg/LightModel>
#include <osg/Texture2D>
#include <osgUtil/SmoothingVisitor>

#include <osgbDynamics/GroundPlane.h>

#include <osgbInteraction/LaunchHandler.h>


#include <osgwTools/Shapes.h>


#include <BulletSoftBody/btSoftBodyHelpers.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>
#include <BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h>
#include <BulletSoftBody/btSoftBody.h>

#include "btBulletDynamicsCommon.h"
#include "CommonInterfaces/CommonRigidBodyBase.h"
#include "OpenGLWindow/ShapeData.h"



#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coNavInteraction.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coMouseButtonInteraction.h>

#include <string>
namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coPotiMenuItem;
class coMouseButtonInteraction;
}
using namespace vrui;
class Brickwall : public opencover::coVRPlugin
{
public:
    coMouseButtonInteraction *InteractionA;
    void preFrame() override;
    class CommonExampleInterface* BasicExampleCreateFunc(struct CommonExampleOptions& options);
    Brickwall();
    ~Brickwall();
    bool init();
    bool update();
    btSoftRigidDynamicsWorld* initPhysics();
    osg::Transform* createOSGBox(osg::Vec3 size, osg::Vec4 rotation, osg::Vec4 color);

    osg::Node* createCube(const osg::Vec3& center, btSoftRigidDynamicsWorld* dw);
    osg::Transform* createDynamicBall(float mass, float radius, const osg::Vec3& center, btDynamicsWorld* dw);
    osg::Transform* createBall(float radius);
    osg::Node* createGround(float w, float h, const osg::Vec3& center, btSoftRigidDynamicsWorld* dw);
    osg::Transform* makeBrick(const osg::Vec3 size, osg::Vec4 rotation, osg::Vec4 color);

    btSoftRigidDynamicsWorld* bulletWorld;
   
    osgbDynamics::MotionState* motionState;
    osg::Group* root;
    btSoftBody* softBody;
    osg::Geometry *geom;
    osg::ref_ptr< osg::Node > platform;
    int moveY;
    int resX;
    int resY;
    unsigned int size;
    double length;
    double radius;
    btSoftRigidDynamicsWorld* bw;
  
};
#endif
