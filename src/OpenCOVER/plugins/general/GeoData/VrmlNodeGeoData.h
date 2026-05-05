/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 2001 Uwe Woessner
//
//  %W% %G%
//  VrmlNodeGeoData.h

#ifndef _VRMLNODEGeoData_
#define _VRMLNODEGeoData_

#include <util/coTypes.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

using namespace vrml;

class VrmlNodeGeoData : public VrmlNodeChild
{
public:
    VrmlNodeGeoData(VrmlScene *scene = 0);
    VrmlNodeGeoData(const VrmlNodeGeoData &n);

    static void initFields(VrmlNodeGeoData *node, vrml::VrmlNodeType *t);
    static const char *typeName();

private:
    VrmlSFVec3f d_offset;
    VrmlSFString d_offsetName;
};
#endif //_VRMLNODEGeoData_
