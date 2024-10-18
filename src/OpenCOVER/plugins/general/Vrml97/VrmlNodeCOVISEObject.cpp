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
#include <util/unixcompat.h>
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

void VrmlNodeCOVISEObject::initFields(VrmlNodeCOVISEObject *node, vrml::VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("objectName", node->d_objectName));
}

const char *VrmlNodeCOVISEObject::name()
{
    return "COVISEObject";
}

VrmlNodeCOVISEObject::VrmlNodeCOVISEObject(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_objectName(NULL)
{
    d_viewerObject = 0;
    group = new osg::Group();
    group->setName(name());
    COVISEObjectNodes.push_back(this);
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeCOVISEObject::VrmlNodeCOVISEObject(const VrmlNodeCOVISEObject &n)
    : VrmlNodeChild(n)
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
