/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeAnchor.cpp
//

#include "VrmlNodeAnchor.h"
#include "VrmlNodeType.h"

#include "VrmlScene.h"

using namespace vrml;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeAnchor(scene);
}

// Define the built in VrmlNodeType:: "Anchor" fields

void VrmlNodeAnchor::initFields(VrmlNodeAnchor *node, VrmlNodeType *t)
{
    VrmlNodeGroup::initFields(node, t);
    initFieldsHelper(node, t,
                     exposedField("description", node->d_description),
                     exposedField("parameter", node->d_parameter),
                     exposedField("url", node->d_url));
}

const char *VrmlNodeAnchor::name() { return "Anchor"; }

VrmlNodeAnchor::VrmlNodeAnchor(VrmlScene *scene)
    : VrmlNodeGroup(scene, name())
{
}

VrmlNodeAnchor::VrmlNodeAnchor(const VrmlNodeAnchor &n)
    : VrmlNodeGroup(n)
{
    d_description = n.d_description;
    d_parameter = n.d_parameter;
    d_url = n.d_url;
}

void VrmlNodeAnchor::render(Viewer *viewer)
{
    viewer->beginObject(name(), 0, this);

    // Render children
    VrmlNodeGroup::render(viewer);
    viewer->setSensitive(this);

    viewer->setSensitive(0);
    viewer->endObject();
}

// Handle a click by loading the url

void VrmlNodeAnchor::activate()
{
    if (d_scene && d_url.size() > 0)
    {

        d_scene->queueLoadUrl(&d_url, &d_parameter);
        //if (! d_scene->loadUrl( &d_url, &d_parameter ))
        //System::the->warn("Couldn't load URL %s\n", d_url[0]);
    }
}

bool VrmlNodeAnchor::isOnlyGeometry() const
{
    return false;
}
