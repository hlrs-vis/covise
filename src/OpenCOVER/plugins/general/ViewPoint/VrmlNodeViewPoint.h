/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeViewPoint.h

#ifndef _VrmlNodeViewPoint_
#define _VrmlNodeViewPoint_

#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

using namespace vrml;

#define ALL_REGIONS_STRING "all"

class VrmlNodeViewPoint : public VrmlNodeChild
{
public:
    VrmlNodeViewPoint(VrmlScene *scene = 0);
    VrmlNodeViewPoint(const VrmlNodeViewPoint &n);

    static void initFields(VrmlNodeViewPoint *node, vrml::VrmlNodeType *t);
    static const char *typeName();

private:
    VrmlSFFloat d_transitionDuration;
    VrmlSFString d_viewPointName;
};
#endif //_VrmlNodeViewPoint_
