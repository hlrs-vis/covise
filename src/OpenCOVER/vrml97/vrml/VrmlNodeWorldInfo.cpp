/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeWorldInfo.cpp

#include "VrmlNodeWorldInfo.h"

#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "System.h"

using namespace vrml;

//  WorldInfo factory.

static VrmlNode *creator(VrmlScene *s)
{
    return new VrmlNodeWorldInfo(s);
}

// Define the built in VrmlNodeType:: "WorldInfo" fields

VrmlNodeType *VrmlNodeWorldInfo::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("WorldInfo", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addField("info", VrmlField::MFSTRING);
    t->addField("title", VrmlField::SFSTRING);
    t->addField("correctBackFaceCulling", VrmlField::SFBOOL);
    t->addField("correctSpatializedAudio", VrmlField::SFBOOL);

    return t;
}

VrmlNodeType *VrmlNodeWorldInfo::nodeType() const { return defineType(0); }

VrmlNodeWorldInfo::VrmlNodeWorldInfo(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_correctBackFaceCulling(true)
    , d_correctSpatializedAudio(true)
{

    if (System::the->getConfigState("COVER.Plugin.Vrml97.CorrectBackfaceCulling", true))
    {
        d_correctBackFaceCulling = true;
    }
    else
    {
        d_correctBackFaceCulling = false;
    }
    System::the->enableCorrectBackFaceCulling(d_correctBackFaceCulling.get());
    System::the->enableCorrectSpatializedAudio(d_correctSpatializedAudio.get());
}

VrmlNodeWorldInfo::~VrmlNodeWorldInfo()
{
}

VrmlNode *VrmlNodeWorldInfo::cloneMe() const
{
    return new VrmlNodeWorldInfo(*this);
}

std::ostream &VrmlNodeWorldInfo::printFields(std::ostream &os, int indent)
{
    if (d_title.get())
        PRINT_FIELD(title);
    if (d_info.size() > 0)
        PRINT_FIELD(info);
    if (!d_correctBackFaceCulling.get())
        PRINT_FIELD(correctBackFaceCulling);
    if (!d_correctSpatializedAudio.get())
        PRINT_FIELD(correctSpatializedAudio);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeWorldInfo::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(info, MFString)
    else if
        TRY_FIELD(title, SFString)
    else if
        TRY_FIELD(correctBackFaceCulling, SFBool)
    else if
        TRY_FIELD(correctSpatializedAudio, SFBool)
    else
        VrmlNode::setField(fieldName, fieldValue);

    if (strcmp(fieldName, "correctBackFaceCulling") == 0)
    {
        System::the->enableCorrectBackFaceCulling(d_correctBackFaceCulling.get());
    }
    else if (strcmp(fieldName, "correctSpatializedAudio") == 0)
    {
        System::the->enableCorrectSpatializedAudio(d_correctSpatializedAudio.get());
    }
}

const VrmlField *VrmlNodeWorldInfo::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "info") == 0)
        return &d_info;
    else if (strcmp(fieldName, "title") == 0)
        return &d_title;
    else if (strcmp(fieldName, "correctBackFaceCulling") == 0)
        return &d_correctBackFaceCulling;
    else if (strcmp(fieldName, "correctSpatializedAudio") == 0)
        return &d_correctSpatializedAudio;

    return VrmlNode::getField(fieldName);
}
