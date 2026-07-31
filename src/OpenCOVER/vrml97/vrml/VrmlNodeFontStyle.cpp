/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeFontStyle.cpp

#include "VrmlNodeFontStyle.h"
#include "VrmlNodeType.h"
#include "MathUtils.h"

using namespace vrml;

void VrmlNodeFontStyle::initFields(VrmlNodeFontStyle *node, VrmlNodeType *t)
{
    VrmlNode::initFieldsHelper(node, t,
                                       exposedField("family", &VrmlNodeFontStyle::d_family),
                                       exposedField("horizontal", &VrmlNodeFontStyle::d_horizontal),
                                       exposedField("justify", &VrmlNodeFontStyle::d_justify),
                                       exposedField("language", &VrmlNodeFontStyle::d_language),
                                       exposedField("leftToRight", &VrmlNodeFontStyle::d_leftToRight),
                                       exposedField("size", &VrmlNodeFontStyle::d_size),
                                       exposedField("spacing", &VrmlNodeFontStyle::d_spacing),
                                       exposedField("style", &VrmlNodeFontStyle::d_style),
                                       exposedField("topToBottom", &VrmlNodeFontStyle::d_topToBottom));
}

const char *VrmlNodeFontStyle::typeName() { return "FontStyle"; }


VrmlNodeFontStyle::VrmlNodeFontStyle(VrmlScene *scene)
    : VrmlNode(scene, typeName())
    , d_family("SERIF")
    , d_horizontal(true)
    , d_justify("BEGIN")
    , d_leftToRight(true)
    , d_size(1.0)
    , d_spacing(1.0)
    , d_style("PLAIN")
    , d_topToBottom(true)
{
}
