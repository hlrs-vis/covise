/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeIndexedSet.h

#ifndef _VRMLNODEINDEXEDSET_
#define _VRMLNODEINDEXEDSET_

#include "VrmlNodeColoredSet.h"

#include "VrmlMFInt.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeIndexedSet : public VrmlNodeColoredSet
{

public:
    // Define the fields of indexed face set nodes
    static VrmlNodeType *defineType(VrmlNodeType *t);

    VrmlNodeIndexedSet(VrmlScene *);
    virtual ~VrmlNodeIndexedSet();

    virtual bool isModified() const;

    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual const VrmlMFInt &getCoordIndex() const;
    virtual const VrmlMFInt &getColorIndex() const;

protected:
    VrmlMFInt d_colorIndex;
    VrmlMFInt d_coordIndex;
};
}
#endif //_VRMLNODEINDEXEDSET_
