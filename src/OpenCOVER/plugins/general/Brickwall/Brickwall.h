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

#include <cover/ui/Owner.h>
#include <osg/ref_ptr>
#include <osg/Node>
#include <string>

class btSoftRigidDynamicsWorld;
class btSoftBody;
class btDynamicsWorld;
class btRigidBody;

namespace osgbDynamics
{
    class MotionState;
}
namespace opencover
{
    namespace ui
    {
        class Menu;
        class Button;
    }
}
namespace osg
{
    class Transform;
    class Group;
	class Geometry;
}
namespace vrui
{
	class coCombinedButtonInteraction;
}
#define NUM_BRICKS 35

class Brickwall : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    class CommonExampleInterface* BasicExampleCreateFunc(struct CommonExampleOptions& options);
    Brickwall();
    ~Brickwall();
    bool init();
    bool update();
 
private:
    vrui::coCombinedButtonInteraction* InteractionA;

    opencover::ui::Menu* BrickTab = nullptr;

    opencover::ui::Button* enableShooting = nullptr;
    btSoftRigidDynamicsWorld* initPhysics();
    osg::Transform* createOSGBox(osg::Vec3 size, osg::Vec4 rotation, osg::Vec4 color);
    
    osg::Node* createCube(const osg::Vec3& center, btDynamicsWorld* dw);
    osg::Transform* createDynamicBall(float mass, float radius, const osg::Vec3& center, btDynamicsWorld* dw, const osg::Vec3& direction);
    osg::Transform* createBall(float radius);
    osg::Node* createGround(float w, float h, const osg::Vec3& center, btDynamicsWorld* dw);
    osg::Transform* makeBrick(const osg::Vec3 size, osg::Vec4 rotation, osg::Vec4 color);
    btRigidBody* groundBody;
    btSoftRigidDynamicsWorld* bulletWorld;
   
    osgbDynamics::MotionState* motionState;
    osg::Group* root;
    btSoftBody* softBody;
    osg::Geometry *geom;
    osg::Vec3 centerOM;
    osg::ref_ptr< osg::Node > platform;
    int moveY;
    int resX;
    int resY;
    int index;
    
    unsigned int size;
    double length;
    double radius;
    btSoftRigidDynamicsWorld* bw;
    osg::ref_ptr< osg::Node > ground;
    osg::ref_ptr< osg::Node > bricks;
    osg::Transform* ball;
    btRigidBody* brickBody;
    btRigidBody* ballBody;
};
#endif
