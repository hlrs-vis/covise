/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeCrawler.h

#ifndef _VRMLNODECrawler_
#define _VRMLNODECrawler_

#include <util/coTypes.h>
#include "PxPhysicsAPI.h"

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

class CrawlerConnection;

using namespace opencover;
using namespace vrml;
using namespace physx;
namespace opencover
{
    
//------------------------------------------------------------------------------------------------------------------------------

class Flap
{
public:
    PxVec3 verts[6];
    PxConvexMesh* convexMesh;
    PxShape* Shape;
    PxRigidDynamic* Actor;
    PxRevoluteJoint* revoluteJoint;
};

}

class VrmlNodeCrawler : public VrmlNodeChild
{

public:
    // Define the fields of Crawler nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeCrawler(VrmlScene *scene = 0);
    VrmlNodeCrawler(const VrmlNodeCrawler &n);
    virtual ~VrmlNodeCrawler();
    virtual void addToScene(VrmlScene *s, const char *);

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeCrawler *toCrawler() const;

    virtual ostream &printFields(ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual void render(Viewer *);


    void CloseFlap();
    void OpenFlap();
    void CrawlerLademodus();


private:
    // Fields
    
    PxRigidDynamic*				tetraederActor;

    // Actor Joints
    PxRevoluteJoint*			revoluteJointBMF;
    PxRevoluteJoint*			revoluteJointBSF;

    // Selected Actor
    PxRigidDynamic*				gSelectedActorDynamic;


    Flap MainFlap[3];
    Flap SecFlap[3];
    VrmlSFFloat d_speed;
    VrmlSFBool d_lademodus;
    VrmlSFVec3f d_position;
    VrmlSFRotation d_rotation;
    VrmlSFFloat d_main0Angle;
    VrmlSFFloat d_main1Angle;
    VrmlSFFloat d_main2Angle;
    VrmlSFFloat d_sec0Angle;
    VrmlSFFloat d_sec1Angle;
    VrmlSFFloat d_sec2Angle;
};
#endif //_VRMLNODECrawler_
