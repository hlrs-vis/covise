/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeVariant.cpp
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
#include <vrml97/vrml/coEventQueue.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/VRViewer.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRPluginSupport.h>
#include <math.h>
#include "VrmlNodeVariant.h"
#include "VariantPlugin.h"

VrmlNodeVariant *VrmlNodeVariant::theVariantNode=NULL;
VrmlNodeVariant *VrmlNodeVariant::instance()
{
    if (System::the)
    {
        if(theVariantNode==NULL)
        {
            theVariantNode = new VrmlNodeVariant();
            initFields(theVariantNode, nullptr);
        }
        return theVariantNode;
    }
    return nullptr;
}

void VrmlNodeVariant::initFields(VrmlNodeVariant *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("variant", node->d_variant, [](auto f){
                        VariantPlugin::plugin->setVariant(f->get());
                     }));
}

const char *VrmlNodeVariant::name() { return "Variant"; }

VrmlNodeVariant::VrmlNodeVariant(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , d_variant("none")
{
    setModified();
}

void VrmlNodeVariant::addToScene(VrmlScene *s, const char *relUrl)
{
    (void)relUrl;
    d_scene = s;
    if (s)
    {
    }
    else
    {
        cerr << "no Scene" << endl;
    }
}

// need copy constructor for new markerName (each instance definitely needs a new marker Name) ...

VrmlNodeVariant::VrmlNodeVariant(const VrmlNodeVariant &n)
    : VrmlNodeChild(n)
    , d_variant(n.d_variant)
{
    setModified();
}

VrmlNodeVariant::~VrmlNodeVariant()
{
    if (this == theVariantNode) {
        theVariantNode = nullptr;
    }
}

VrmlNodeVariant *VrmlNodeVariant::toVariant() const
{
    return (VrmlNodeVariant *)this;
}

void VrmlNodeVariant::render(Viewer *viewer)
{
    (void)viewer;
}

void VrmlNodeVariant::setVariant(std::string varName)
{
    d_variant.set(varName.c_str());
}
