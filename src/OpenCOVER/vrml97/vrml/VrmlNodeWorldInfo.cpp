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

void VrmlNodeWorldInfo::initFields(VrmlNodeWorldInfo *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t,
                     field("info", node->d_info),
                     field("title", node->d_title),
                     field("correctBackFaceCulling", node->d_correctBackFaceCulling, [](auto correctBackFaceCulling) {
                         System::the->enableCorrectBackFaceCulling(correctBackFaceCulling->get());
                     }),
                     field("correctSpatializedAudio", node->d_correctSpatializedAudio, [](auto correctSpatializedAudio) {
                         System::the->enableCorrectSpatializedAudio(correctSpatializedAudio->get());
                     }));
}

const char *VrmlNodeWorldInfo::name() { return "WorldInfo"; }

VrmlNodeWorldInfo::VrmlNodeWorldInfo(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
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
