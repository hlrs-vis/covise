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

#ifndef _VRMLNODECOLORRGBA_
#define _VRMLNODECOLORRGBA_

#include "VrmlNode.h"
#include "VrmlMFColorRGBA.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeColorRGBA : public VrmlNode
{

public:
    // Define the fields of Color nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeColorRGBA(VrmlScene *);
    virtual ~VrmlNodeColorRGBA();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeColorRGBA *toColorRGBA() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    VrmlMFColorRGBA &color()
    {
        return d_color;
    }

private:
    VrmlMFColorRGBA d_color;
};
}
#endif //_VRMLNODECOLORRGBA_
