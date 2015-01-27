/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodePhysics.h

#ifndef _VRMLNODEPhysics_
#define _VRMLNODEPhysics_

#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlScene.h>
#include <plugins/general/Vrml97/ViewerOsg.h>

#include <osgbBullet/MotionState.h>
#include <osgbBullet/CollisionShapes.h>
#include <osgbBullet/Utils.h>

#include <btBulletDynamicsCommon.h>

using namespace covise;
using namespace opencover;

namespace covise
{

class VrmlNodePhysics : public VrmlNodeTransform
{

public:
    // Define the fields of Physics nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodePhysics(VrmlScene *scene = 0);
    VrmlNodePhysics(const VrmlNodePhysics &n);
    virtual ~VrmlNodePhysics();
    //virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodePhysics *toPhysics() const;

    virtual ostream &printFields(ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual void render(Viewer *);

    bool isEnabled()
    {
        return d_enabled.get();
    }

    static btDynamicsWorld *bw;

private:
    // Fields

    VrmlSFBool d_enabled;
    VrmlSFFloat d_mass;
    VrmlSFVec3f d_inertia;
    btCollisionShape *cs;
    osgbBullet::MotionState *motion;
    btRigidBody *body;
    osg::MatrixTransform *transformNode;
};
}

#endif //_VRMLNODEPhysics_
