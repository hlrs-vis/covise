/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeColor.h

#ifndef _VRMLNODECOLOR_
#define _VRMLNODECOLOR_

#include "VrmlNode.h"
#include "VrmlMFColor.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeColor : public VrmlNode
{

public:
    // Define the fields of Color nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeColor(VrmlScene *);
    virtual ~VrmlNodeColor();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeColor *toColor() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    VrmlMFColor &color()
    {
        return d_color;
    }

private:
    VrmlMFColor d_color;
};
}
#endif //_VRMLNODECOLOR_
