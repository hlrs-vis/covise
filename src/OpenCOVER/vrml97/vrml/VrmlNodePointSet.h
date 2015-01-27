/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodePointSet.h

#ifndef _VRMLNODEPOINTSET_
#define _VRMLNODEPOINTSET_

#include "VrmlNodeGeometry.h"
#include "VrmlSFNode.h"

namespace vrml
{

class VRMLEXPORT VrmlNodePointSet : public VrmlNodeGeometry
{

public:
    // Define the fields of pointSet nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodePointSet(VrmlScene *);
    virtual ~VrmlNodePointSet();

    virtual VrmlNode *cloneMe() const;
    virtual void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;

    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual Viewer::Object insertGeometry(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

protected:
    VrmlSFNode d_color;
    VrmlSFNode d_coord;
};
}
#endif //_VRMLNODEPOINTSET_
