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
    VrmlNodeTemplate::initFieldsHelper(node, t,
                                       exposedField("family", node->d_family),
                                       exposedField("horizontal", node->d_horizontal),
                                       exposedField("justify", node->d_justify),
                                       exposedField("language", node->d_language),
                                       exposedField("leftToRight", node->d_leftToRight),
                                       exposedField("size", node->d_size),
                                       exposedField("spacing", node->d_spacing),
                                       exposedField("style", node->d_style),
                                       exposedField("topToBottom", node->d_topToBottom));
}

const char *VrmlNodeFontStyle::name() { return "FontStyle"; }


VrmlNodeFontStyle::VrmlNodeFontStyle(VrmlScene *scene)
    : VrmlNodeTemplate(scene, name())
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

VrmlNodeFontStyle *VrmlNodeFontStyle::toFontStyle() const
{
    return (VrmlNodeFontStyle *)this;
}
