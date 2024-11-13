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
    static void initFields(VrmlNodeMetadataSet *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeMetadataSet(VrmlScene *);
    virtual void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;
    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relativeUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

protected:
    VrmlMFNode d_value;
};
}
#endif // _VRMLNODEMETADATASET_
