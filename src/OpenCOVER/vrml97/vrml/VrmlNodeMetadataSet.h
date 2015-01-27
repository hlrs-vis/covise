/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataSet.h

#ifndef _VRMLNODEMetadataSet_
#define _VRMLNODEMetadataSet_

#include "VrmlNodeMetadata.h"
#include "VrmlMFNode.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeMetadataSet : public VrmlNodeMetadata

{

public:
    // Define the fields of indexed face set nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeMetadataSet(VrmlScene *);
    virtual ~VrmlNodeMetadataSet();

    virtual VrmlNode *cloneMe() const;
    virtual void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;
    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relativeUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual VrmlNodeMetadataSet *toMetadataSet() const;

protected:
    VrmlMFNode d_value;
};
}
#endif // _VRMLNODEMETADATASET_
