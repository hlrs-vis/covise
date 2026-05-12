/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRML_NODE_SKY_H
#define _VRML_NODE_SKY_H

#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlNodeChild.h>

using namespace vrml;

class VrmlNodeSky : public VrmlNodeChild
{
public:
    VrmlNodeSky(VrmlScene *scene = 0);
    VrmlNodeSky(const VrmlNodeSky &n);

    static void initFields(VrmlNodeSky *node, vrml::VrmlNodeType *t);
    static const char *typeName();

private:
    VrmlSFString d_skyName = "";
    VrmlSFFloat d_top = 0.5;
    VrmlSFFloat d_bottom = 0.48;
    VrmlSFColor d_floorColor = { 0.0, 0.0, 0.0 };
};
#endif
