/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _osgBulletPlugin_PLUGIN_H
#define _osgBulletPlugin_PLUGIN_H



#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osg/Geode>
#include <osg/LightModel>
#include <osg/Texture2D>
#include <osgUtil/SmoothingVisitor>

#include <osgbDynamics/GroundPlane.h>
#include <osgbCollision/GLDebugDrawer.h>
#include <osgbCollision/Utils.h>
#include <osgbInteraction/DragHandler.h>
#include <osgbInteraction/LaunchHandler.h>
#include <osgbInteraction/SaveRestoreHandler.h>

#include <osgwTools/Shapes.h>

#include <btBulletDynamicsCommon.h>
#include <BulletSoftBody/btSoftBodyHelpers.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>
#include <BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h>
#include <BulletSoftBody/btSoftBody.h>

#include <osg/io_utils>
#include <string>

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: osgBulletPlugin OpenCOVER Plugin (is polite)                          **
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

class osgBulletPlugin : public opencover::coVRPlugin
{
public:
    
    osgBulletPlugin();
    ~osgBulletPlugin();
    bool init();
    bool update();
    btSoftBody *softBody;
    unsigned int size;
    osg::Node* makeFlag(btSoftRigidDynamicsWorld* bw);
    osg::Geometry *geom;
    int resX;
    int resY;
    btSoftRigidDynamicsWorld* bw;
  

  
};
#endif
