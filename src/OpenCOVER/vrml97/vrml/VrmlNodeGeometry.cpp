/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeGeometry.cpp

#include "VrmlNodeGeometry.h"
#include "VrmlNodeType.h"

using namespace vrml;


void VrmlNodeGeometry::initFields(VrmlNodeGeometry *node, VrmlNodeType *t)
{
    //space for future implementations
}

VrmlNodeGeometry::VrmlNodeGeometry(VrmlScene *s, const std::string &name)
    : VrmlNodeTemplate(s, name)
    , d_viewerObject(0)
{
}

VrmlNodeGeometry *VrmlNodeGeometry::toGeometry() const
{
    return (VrmlNodeGeometry *)this;
}

VrmlNodeColor *VrmlNodeGeometry::color() { return 0; }

bool VrmlNodeGeometry::isOnlyGeometry() const
{
    if (!VrmlNode::isOnlyGeometry())
        return false;

    return true;
}

// Geometry nodes need only define insertGeometry(), not render().

void VrmlNodeGeometry::render(Viewer *v)
{
    if (d_viewerObject && isModified())
    {
        v->removeObject(d_viewerObject);
        d_viewerObject = 0;
    }

    if (d_viewerObject)
        v->insertReference(d_viewerObject);
    else
    {
        d_viewerObject = insertGeometry(v);
        clearModified();
    }
}
