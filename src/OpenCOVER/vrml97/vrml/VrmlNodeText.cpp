/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeText.cpp

#include "VrmlNodeText.h"

#include "VrmlNodeType.h"
#include "VrmlNodeFontStyle.h"
#include "MathUtils.h"
#include "Viewer.h"

using namespace vrml;

void VrmlNodeText::initFields(VrmlNodeText *node, VrmlNodeType *t)
{
    VrmlNodeGeometry::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
                     exposedField("string", node->d_string),
                     exposedField("fontStyle", node->d_fontStyle),
                     exposedField("length", node->d_length),
                     exposedField("maxExtent", node->d_maxExtent));
}
const char *VrmlNodeText::name() { return "Text"; }

VrmlNodeText::VrmlNodeText(VrmlScene *scene)
    : VrmlNodeGeometry(scene, name())
{
}

void VrmlNodeText::cloneChildren(VrmlNamespace *ns)
{
    if (d_fontStyle.get())
    {
        d_fontStyle.set(d_fontStyle.get()->clone(ns));
        d_fontStyle.get()->parentList.push_back(this);
    }
}

bool VrmlNodeText::isModified() const
{
    return (VrmlNode::isModified() || (d_fontStyle.get() && d_fontStyle.get()->isModified()));
}

void VrmlNodeText::clearFlags()
{
    VrmlNode::clearFlags();
    if (d_fontStyle.get())
        d_fontStyle.get()->clearFlags();
}

void VrmlNodeText::addToScene(VrmlScene *s, const char *relUrl)
{
    nodeStack.push_front(this);
    d_scene = s;
    if (d_fontStyle.get())
        d_fontStyle.get()->addToScene(s, relUrl);
    nodeStack.pop_front();
}

void VrmlNodeText::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);
    if (d_fontStyle.get())
        d_fontStyle.get()->copyRoutes(ns);
    nodeStack.pop_front();
}

Viewer::Object VrmlNodeText::insertGeometry(Viewer *viewer)
{
    char **s = d_string.get();

    if (s)
    {
        int justify[2] = { 1, 1 };
        float size = 1.0;
        VrmlNodeFontStyle *f = 0;
        if (d_fontStyle.get())
            f = d_fontStyle.get()->as<VrmlNodeFontStyle>();

        if (f)
        {
            VrmlMFString &j = f->justify();

            for (int i = 0; i < j.size(); ++i)
            {
                if (strcmp(j[i], "END") == 0)
                    justify[i] = -1;
                else if (strcmp(j[i], "MIDDLE") == 0)
                    justify[i] = 0;
            }
            size = f->size();
        }

        return viewer->insertText(justify, size, d_string.size(), s, name());
    }

    return 0;
}
