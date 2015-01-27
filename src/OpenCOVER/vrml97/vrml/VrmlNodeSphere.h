/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeSphere.h

#ifndef _VRMLNODESPHERE_
#define _VRMLNODESPHERE_

#include "VrmlNodeGeometry.h"
#include "VrmlSFFloat.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeSphere : public VrmlNodeGeometry
{

public:
    // Define the fields of sphere nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeSphere(VrmlScene *);
    virtual ~VrmlNodeSphere();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual Viewer::Object insertGeometry(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual VrmlNodeSphere *toSphere() const; //LarryD Mar 08/99
    virtual float getRadius() //LarryD Mar 08/99
    {
        return d_radius.get();
    }

protected:
    VrmlSFFloat d_radius;
};
}
#endif //_VRMLNODESPHERE_
