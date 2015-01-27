/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TEMPLATE_PLUGIN_H
#define _TEMPLATE_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: Template Plugin (does nothing)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Nov-01  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Geode>

#include <osgbBullet/MotionState.h>
#include <osgbBullet/CollisionShapes.h>
#include <osgbBullet/Utils.h>

#include <btBulletDynamicsCommon.h>

#include <string>
#include <osg/io_utils>

using namespace opencover;

class BulletPlugin : public coVRPlugin
{
public:
    BulletPlugin();
    ~BulletPlugin();

    // this will be called in PreFrame
    void preFrame();

    // this will be called if an object with feedback arrives
    void newInteractor(RenderObject *container, coInteractor *i);

    // this will be called if a COVISE object arrives
    void addObject(RenderObject *container,
                   RenderObject *obj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   osg::Group *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn, float transparency);

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace);

private:
    btDynamicsWorld *bulletWorld;
    osgbBullet::MotionState *shakeMotion;
    osgbBullet::MotionState *pointerMotion;
    btConeShape *pointerShape;
    btRigidBody *pointerBody;
    btDynamicsWorld *initPhysics();
    osg::Geode *osgBox(const osg::Vec3 &center, const osg::Vec3 &halfLengths);
    osg::MatrixTransform *makeDie(btDynamicsWorld *bw);
};
#endif
