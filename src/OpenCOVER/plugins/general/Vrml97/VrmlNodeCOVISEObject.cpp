/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeCOVISEObject.cpp
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
#include "VrmlNodeCOVISEObject.h"
#include "ViewerOsg.h"

list<VrmlNodeCOVISEObject *> VrmlNodeCOVISEObject::COVISEObjectNodes;

// ARSensor factory.

void VrmlNodeCOVISEObject::addNode(osg::Node *node)
{
    list<VrmlNodeCOVISEObject *>::iterator n;
    for (n = COVISEObjectNodes.begin(); n != COVISEObjectNodes.end(); ++n)
    {
        (*n)->addCoviseNode(node);
    }
}

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeCOVISEObject(scene);
}

// Define the built in VrmlNodeType:: "ARSensor" fields

VrmlNodeType *VrmlNodeCOVISEObject::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("COVISEObject", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("objectName", VrmlField::SFSTRING);

    return t;
}

VrmlNodeType *VrmlNodeCOVISEObject::nodeType() const
{
    return defineType(0);
}

VrmlNodeCOVISEObject::VrmlNodeCOVISEObject(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_objectName(NULL)
{
    d_viewerObject = 0;
    group = new osg::Group();
    group->setName(name());
    COVISEObjectNodes.push_back(this);
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeCOVISEObject::VrmlNodeCOVISEObject(const VrmlNodeCOVISEObject &n)
    : VrmlNodeChild(n.d_scene)
    , d_objectName(NULL)
{
    d_viewerObject = 0;
}

VrmlNodeCOVISEObject::~VrmlNodeCOVISEObject()
{
    if (group->getNumParents())
    {
        group->getParent(0)->removeChild(group.get());
    }

    COVISEObjectNodes.remove(this);
}

VrmlNode *VrmlNodeCOVISEObject::cloneMe() const
{
    return new VrmlNodeCOVISEObject(*this);
}

VrmlNodeCOVISEObject *VrmlNodeCOVISEObject::toCOVISEObject() const
{
    return (VrmlNodeCOVISEObject *)this;
}

void VrmlNodeCOVISEObject::render(Viewer *v)
{

    ViewerOsg *viewer = (ViewerOsg *)v;

    if (d_viewerObject)
        viewer->insertReference(d_viewerObject);
    else
    {
        d_viewerObject = viewer->beginObject(name(), 0, this);
        viewer->insertNode(group.get());
        viewer->endObject();
    }

    clearModified();
}

ostream &VrmlNodeCOVISEObject::printFields(ostream &os, int indent)
{
    if (!d_objectName.get())
        PRINT_FIELD(objectName);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeCOVISEObject::setField(const char *fieldName,
                                    const VrmlField &fieldValue)
{
    if
        TRY_FIELD(objectName, SFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeCOVISEObject::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "objectName") == 0)
        return &d_objectName;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeCOVISEObject::addCoviseNode(osg::Node *node)
{
    int nLen = strlen(d_objectName.get());
    if ((strcasecmp(d_objectName.get(), "all") == 0 || strncmp(node->getName().c_str(), d_objectName.get(), nLen) == 0))
    {
        // this is the object we are looking for.
        if (node->getNumParents())
        {
            osg::Group *g = node->getParent(0);
            group->addChild(node);
            g->removeChild(node);
        }
        else
        {
            group->addChild(node);
        }
    }
}
