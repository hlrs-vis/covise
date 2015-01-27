/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBox.h

#ifndef _VRMLNODEBOX_
#define _VRMLNODEBOX_

#include "VrmlNodeGeometry.h"
#include "VrmlSFVec3f.h"

namespace vrml
{

class VRMLEXPORT VrmlScene;

class VRMLEXPORT VrmlNodeBox : public VrmlNodeGeometry
{

public:
    // Define the fields of box nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeBox(VrmlScene *);
    virtual ~VrmlNodeBox();

    virtual VrmlNode *cloneMe() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual Viewer::Object insertGeometry(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    virtual VrmlNodeBox *toBox() const; //LarryD Mar 08/99
    virtual const VrmlSFVec3f &getSize() const; //LarryD Mar 08/99

protected:
    VrmlSFVec3f d_size;
};
}
#endif //_VRMLNODEBOX_
