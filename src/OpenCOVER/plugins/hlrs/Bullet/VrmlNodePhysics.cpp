/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodePhysics.cpp
#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <util/common.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <math.h>

#include <util/byteswap.h>

#include "VrmlNodePhysics.h"
//#include "ViewerOsg.h"
#include <osg/MatrixTransform>
#include <osg/Quat>

// Physics factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodePhysics(scene);
}

btDynamicsWorld *VrmlNodePhysics::bw = NULL;

// Define the built in VrmlNodeType:: "Physics" fields

VrmlNodeType *VrmlNodePhysics::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Physics", creator);
    }

    VrmlNodeTransform::defineType(t); // Parent class
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addExposedField("mass", VrmlField::SFFLOAT);
    t->addExposedField("inertia", VrmlField::SFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodePhysics::nodeType() const
{
    return defineType(0);
}

VrmlNodePhysics::VrmlNodePhysics(VrmlScene *scene)
    : VrmlNodeTransform(scene)
    , d_enabled(true)
    , d_mass(0)
    , d_inertia(0, 0, 0)
{
    setModified();
    cs = NULL;
    motion = NULL;
    body = NULL;
    transformNode = NULL;
}

/*
void VrmlNodePhysics::addToScene(VrmlScene *s, const char *relUrl)
{
(void)relUrl;
d_scene = s;
if(s)
{
//addARNode(this);
}
else
{
cerr << "no Scene" << endl;
}
}*/

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodePhysics::VrmlNodePhysics(const VrmlNodePhysics &n)
    : VrmlNodeTransform(n.d_scene)
    , d_enabled(n.d_enabled)
    , d_mass(n.d_mass)
    , d_inertia(n.d_inertia)
{
    setModified();
    cs = NULL;
    motion = NULL;
    body = NULL;
    transformNode = NULL;
}

VrmlNodePhysics::~VrmlNodePhysics()
{
}

VrmlNode *VrmlNodePhysics::cloneMe() const
{
    return new VrmlNodePhysics(*this);
}

VrmlNodePhysics *VrmlNodePhysics::toPhysics() const
{
    return (VrmlNodePhysics *)this;
}

void VrmlNodePhysics::render(Viewer *viewer)
{
    bool notInSceneGraph = false;
    if (!haveToRender())
        return;

    if (d_xformObject && isModified())
    {
        viewer->removeObject(d_xformObject);
        d_xformObject = 0;
    }
    checkAndRemoveNodes(viewer);
    if (d_xformObject)
    {
        viewer->insertReference(d_xformObject);
    }
    else if (d_children.size() > 0)
    {
        d_xformObject = viewer->beginObject(name(), 0, this);

        // Apply transforms
        viewer->setTransform(d_center.get(),
                             d_rotation.get(),
                             d_scale.get(),
                             d_scaleOrientation.get(),
                             d_translation.get(), d_modified);

        // Render children
        VrmlNodeGroup::render(viewer);

        if (motion == NULL)
        {
            osgViewerObject *vo = (osgViewerObject *)d_xformObject;
            transformNode = (osg::MatrixTransform *)vo->pNode.get();

            cs = osgbBullet::btBoxCollisionShapeFromOSG(transformNode);
            motion = new osgbBullet::MotionState();
            motion->setTransform(transformNode);
            cerr << "masse: " << name() << " " << d_mass.get() << endl;
            btScalar mass(d_mass.get());
            btVector3 inertia(d_inertia.x(), d_inertia.y(), d_inertia.z());
            cs->calculateLocalInertia(mass, inertia);
            btRigidBody::btRigidBodyConstructionInfo rb(mass, motion, cs, inertia);
            body = new btRigidBody(rb);
            //body->setActivationState( DISABLE_DEACTIVATION );
            bw->addRigidBody(body);
            //motion->setWorldTransform( osgbBullet::asBtTransform(transformNode->getMatrix()) );
            notInSceneGraph = true;
        }
        else
        {

            osg::Node *currentNode, * or = cover->getObjectsRoot();
            osg::Matrix geoToWC, tmpMat;
            currentNode = transformNode->getParent(0);
            //currentNode = transformNode;
            geoToWC.makeIdentity();
            while (currentNode != NULL && currentNode != or )
            {
                if (dynamic_cast<osg::MatrixTransform *>(currentNode))
                {
                    tmpMat = ((osg::MatrixTransform *)currentNode)->getMatrix();
                    geoToWC.postMult(tmpMat);
                }
                if (currentNode->getNumParents() > 0)
                    currentNode = currentNode->getParent(0);
                else
                    currentNode = NULL;
            }
            //motion->setParentTransform( geoToWC );
            motion->setParentTransform(transformNode->getMatrix());
            motion->setWorldTransform(osgbBullet::asBtTransform(geoToWC));
        }

        // Reverse transforms (for immediate mode/no matrix stack renderer)
        viewer->unsetTransform(d_center.get(),
                               d_rotation.get(),
                               d_scale.get(),
                               d_scaleOrientation.get(),
                               d_translation.get());
        viewer->endObject();
    }
    /*if(d_mass.get()==0.0)
   {
   clearModified();
   }
   else
   {
   setModified();
   }
   */
    if (notInSceneGraph)
        setModified();
    else
        clearModified();
}

ostream &VrmlNodePhysics::printFields(ostream &os, int indent)
{
    if (!d_enabled.get())
        PRINT_FIELD(enabled);
    if (!d_mass.get())
        PRINT_FIELD(mass);
    if (!d_inertia.get())
        PRINT_FIELD(inertia);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodePhysics::setField(const char *fieldName,
                               const VrmlField &fieldValue)
{
    if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(mass, SFFloat)
    else if
        TRY_FIELD(inertia, SFVec3f)
    else
        VrmlNodeTransform::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodePhysics::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "mass") == 0)
        return &d_mass;
    else if (strcmp(fieldName, "inertia") == 0)
        return &d_inertia;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
