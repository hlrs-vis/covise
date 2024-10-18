/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeClippingPlane.cpp
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
#include <PluginUtil/PluginMessageTypes.h>
#include <math.h>

#include "VrmlNodeClippingPlane.h"
#include "ViewerOsg.h"
#include <osg/MatrixTransform>
#include <osg/Quat>


void VrmlNodeClippingPlane::initFields(VrmlNodeClippingPlane *node, vrml::VrmlNodeType *t)
{
    VrmlNodeGroup::initFields(node, t);
    initFieldsHelper(node, t,
        exposedField("global", node->d_global),
        exposedField("enabled", node->d_enabled),
        exposedField("position", node->d_position),
        exposedField("orientation", node->d_orientation),
        exposedField("number", node->d_number));
}

const char *VrmlNodeClippingPlane::name()
{
    return "ClippingPlane";
}

VrmlNodeClippingPlane::VrmlNodeClippingPlane(VrmlScene *scene)
    : VrmlNodeGroup(scene, name())
    , d_global(false)
    , d_enabled(true)
    , d_position(0, 0, 0)
    , d_orientation(0, 1, 0, 0)
    , d_number(0)
{
    d_clipObject = 0;
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeClippingPlane::VrmlNodeClippingPlane(const VrmlNodeClippingPlane &n)
    : VrmlNodeGroup(n)
    , d_global(n.d_global)
    , d_enabled(n.d_enabled)
    , d_position(n.d_position)
    , d_orientation(n.d_orientation)
    , d_number(n.d_number)
{
    d_clipObject = 0;
}

void VrmlNodeClippingPlane::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    if (d_clipObject && isModified())
    {
        viewer->removeObject(d_clipObject);
        d_clipObject = 0;
    }
    checkAndRemoveNodes(viewer);
    if (d_clipObject)
    {
        viewer->insertReference(d_clipObject);
    }
    else if (d_children.size() > 0)
    {
        d_clipObject = viewer->beginObject(name(), 0, this);
    }
    if (d_global.get())
    {
        if (d_enabled.get())
        {
            osg::ClipPlane *cp = cover->getClipPlane(d_number.get());
            float *pos = d_position.get();
            float *ori = d_orientation.get();
            osg::Quat q(ori[3], osg::Vec3(ori[0], ori[1], ori[2]));
            osg::Vec3 normal(0, -1, 0);
            normal = q * normal;
            osg::Plane p(normal, osg::Vec3(pos[0], -pos[2], pos[1])); // rotated 90 degrees
            cp->setClipPlane(p);
            cover->getObjectsRoot()->addClipPlane(cp);
        }
        else
        {
            osg::ClipPlane *cp = cover->getClipPlane(d_number.get());
            cover->getObjectsRoot()->removeClipPlane(cp);
        }
    }
    else
    {

        if (d_children.size() > 0)
        {
            // Apply transforms
            viewer->setClip(d_position.get(),
                            d_orientation.get(),
                            d_number.get(),
                            d_enabled.get());
        }
    }
    if (d_children.size() > 0)
    {

        // Render children
        VrmlNodeGroup::render(viewer);
        viewer->endObject();
    }
    clearModified();
}
