/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeShadowedScene.cpp
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

#include "VrmlNodeShadowedScene.h"
#include "ViewerOsg.h"
#include <osg/MatrixTransform>
#include <osg/Quat>

// ARSensor factory.

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeShadowedScene(scene);
}

// Define the built in VrmlNodeType:: "ARSensor" fields

VrmlNodeType *VrmlNodeShadowedScene::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("ShadowedScene", creator);
    }

    VrmlNodeGroup::defineType(t); // Parent class
    t->addExposedField("global", VrmlField::SFBOOL);
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addExposedField("orientation", VrmlField::SFROTATION);
    t->addExposedField("position", VrmlField::SFVEC3F);
    t->addExposedField("number", VrmlField::SFINT32);

    return t;
}

VrmlNodeType *VrmlNodeShadowedScene::nodeType() const
{
    return defineType(0);
}

VrmlNodeShadowedScene::VrmlNodeShadowedScene(VrmlScene *scene)
    : VrmlNodeGroup(scene)
    , d_global(false)
    , d_enabled(true)
    , d_position(0, 0, 0)
    , d_orientation(0, 1, 0, 0)
    , d_number(0)
{
    d_shadowObject = 0;
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeShadowedScene::VrmlNodeShadowedScene(const VrmlNodeShadowedScene &n)
    : VrmlNodeGroup(n.d_scene)
    , d_global(n.d_global)
    , d_enabled(n.d_enabled)
    , d_position(n.d_position)
    , d_orientation(n.d_orientation)
    , d_number(n.d_number)
{
    d_shadowObject = 0;
}

VrmlNodeShadowedScene::~VrmlNodeShadowedScene()
{
}

VrmlNode *VrmlNodeShadowedScene::cloneMe() const
{
    return new VrmlNodeShadowedScene(*this);
}

void VrmlNodeShadowedScene::render(Viewer *viewer)
{
    if (!haveToRender())
        return;

    if (d_shadowObject && isModified())
    {
        viewer->removeObject(d_shadowObject);
        d_shadowObject = 0;
    }
    checkAndRemoveNodes(viewer);
    if (d_shadowObject)
    {
        viewer->insertReference(d_shadowObject);
    }
    else if (d_children.size() > 0)
    {
        d_shadowObject = viewer->beginObject(name(), 0, this);
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
            viewer->setShadow(
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

ostream &VrmlNodeShadowedScene::printFields(ostream &os, int indent)
{
    if (!d_global.get())
        PRINT_FIELD(global);
    if (!d_enabled.get())
        PRINT_FIELD(enabled);
    if (!d_position.get())
        PRINT_FIELD(position);
    if (!d_orientation.get())
        PRINT_FIELD(orientation);
    if (!d_number.get())
        PRINT_FIELD(number);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeShadowedScene::setField(const char *fieldName,
                                     const VrmlField &fieldValue)
{
    if
        TRY_FIELD(global, SFBool)
    else if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(position, SFVec3f)
    else if
        TRY_FIELD(orientation, SFRotation)
    else if
        TRY_FIELD(number, SFInt)
    else
        VrmlNodeGroup::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeShadowedScene::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "global") == 0)
        return &d_global;
    else if (strcmp(fieldName, "position") == 0)
        return &d_position;
    else if (strcmp(fieldName, "orientation") == 0)
        return &d_orientation;
    else if (strcmp(fieldName, "number") == 0)
        return &d_number;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
