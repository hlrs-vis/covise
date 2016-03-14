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
            theVariantNode=new VrmlNodeVariant();
    }
    return theVariantNode;
}



// Variant factory.

static VrmlNode *creator(VrmlScene *scene)
{
    
    VrmlNodeVariant *var = VrmlNodeVariant::instance();
    //var->d_scene = scene;
    return var;
}


// Define the built in VrmlNodeType:: "Variant" fields

VrmlNodeType *VrmlNodeVariant::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Variant", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("variant", VrmlField::SFSTRING);

    return t;
}

VrmlNodeType *VrmlNodeVariant::nodeType() const
{
    return defineType(0);
}

VrmlNodeVariant::VrmlNodeVariant(VrmlScene *scene)
    : VrmlNodeChild(scene)
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
    : VrmlNodeChild(n.d_scene)
    , d_variant(n.d_variant)
{
    setModified();
}

VrmlNodeVariant::~VrmlNodeVariant()
{
}

VrmlNode *VrmlNodeVariant::cloneMe() const
{
    return new VrmlNodeVariant(*this);
}

VrmlNodeVariant *VrmlNodeVariant::toVariant() const
{
    return (VrmlNodeVariant *)this;
}

void VrmlNodeVariant::render(Viewer *viewer)
{
    (void)viewer;
}

ostream &VrmlNodeVariant::printFields(ostream &os, int indent)
{
    if (!d_variant.get())
        PRINT_FIELD(variant);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeVariant::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(variant, SFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);

    if (strcmp(fieldName, "variant") == 0)
    {
        VariantPlugin::plugin->setVariant(fieldValue.toSFString()->get());
    }
}


void VrmlNodeVariant::setVariant(std::string varName)
{
    d_variant.set(varName.c_str());
}

const VrmlField *VrmlNodeVariant::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "variant") == 0)
        return &d_variant;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}
