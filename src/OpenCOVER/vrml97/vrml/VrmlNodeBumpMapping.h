/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
// %W% %G%
//

#ifndef _VRMLNODEBUMPMAPPING_
#define _VRMLNODEBUMPMAPPING_

#include "VrmlNode.h"
#include "VrmlSFFloat.h"
#include "VrmlSFRotation.h"
#include "VrmlSFVec3f.h"
#include "VrmlSFString.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VrmlNodeType;
class VrmlScene;

class VRMLEXPORT VrmlNodeBumpMapping : public VrmlNodeChild
{

public:
    // Define the built in VrmlNodeType:: "BumpMapping"
    static void initFields(VrmlNodeBumpMapping *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeBumpMapping(VrmlScene *);

    virtual void addToScene(VrmlScene *s, const char *relativeUrl);

    virtual void render(Viewer *);

    const VrmlField *getField(const char *fieldName) const;

};
}
#endif //_VRMLNODEBumpMapping__
