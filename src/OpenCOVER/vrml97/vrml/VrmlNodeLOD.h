/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeLOD.h

#ifndef _VRMLNODELOD_
#define _VRMLNODELOD_

#include "VrmlMFNode.h"
#include "VrmlMFFloat.h"
#include "VrmlSFVec3f.h"

#include "VrmlNode.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeLOD : public VrmlNodeChild
{

public:
    // Define the fields of all built in LOD nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeLOD(VrmlScene *);
    virtual ~VrmlNodeLOD();

    virtual VrmlNode *cloneMe() const;
    void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;

    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

    virtual VrmlNodeLOD *toLOD() const;
    VrmlMFNode *getLevel()
    {
        return &d_level;
    }
    virtual const VrmlMFFloat &getRange() const;
    virtual const VrmlSFVec3f &getCenter() const;

protected:
    VrmlMFNode d_level;
    VrmlSFVec3f d_center;
    VrmlMFFloat d_range;
    bool firstTime;
};
}
#endif //_VRMLNODELOD_
