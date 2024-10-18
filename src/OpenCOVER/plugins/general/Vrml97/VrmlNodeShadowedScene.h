/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeShadowedScene.h

#ifndef _VrmlNodeShadowedScene_
#define _VrmlNodeShadowedScene_

#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFVec2f.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlSFRotation.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>
#include <osgShadow/ShadowedScene>

using namespace vrml;
using namespace opencover;

namespace vrml
{

class VRML97COVEREXPORT VrmlNodeShadowedScene : public VrmlNodeGroup
{

public:
    static void initFields(VrmlNodeShadowedScene *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeShadowedScene(VrmlScene *scene = 0);
    VrmlNodeShadowedScene(const VrmlNodeShadowedScene &n);

    void render(Viewer *) override;

private:
    // Fields
    VrmlSFString d_technique;
    VrmlSFNode d_shadowLight;
    VrmlSFFloat d_jitterScale;
    VrmlSFFloat d_softnessWidth;
    VrmlSFVec2f d_textureSize;

    Viewer::Object d_shadowObject;
};
}

#endif //_VrmlNodeShadowedScene_
