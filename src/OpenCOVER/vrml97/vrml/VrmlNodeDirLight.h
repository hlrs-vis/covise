/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeDirLight.h

#ifndef _VRMLNODEDIRLIGHT_
#define _VRMLNODEDIRLIGHT_

#include "VrmlNodeLight.h"
#include "VrmlSFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeDirLight : public VrmlNodeLight
{

public:
    // Define the fields of dirLight nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeDirLight(VrmlScene *);
    virtual ~VrmlNodeDirLight();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName,
                          const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    //LarryD Mar 04/99
    virtual const VrmlSFVec3f &getDirection() const;
    //LarryD Mar 04/99
    virtual VrmlNodeDirLight *toDirLight() const;

protected:
    VrmlSFVec3f d_direction;
};
}
#endif //_VRMLNODEDIRLIGHT_
