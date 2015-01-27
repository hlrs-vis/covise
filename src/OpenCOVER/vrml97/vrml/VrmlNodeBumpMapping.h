/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
// %W% %G%
//

#ifndef _VRMLNODEBUMPMAPPING_
#define _VRMLNODEBUMPMAPPING_

#include "VrmlNode.h"
#include "VrmlSFFloat.h"
#include "VrmlSFRotation.h"
#include "VrmlSFVec3f.h"
#include "VrmlSFString.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VrmlNodeType;
class VrmlScene;

class VRMLEXPORT VrmlNodeBumpMapping : public VrmlNodeChild
{

public:
    // Define the built in VrmlNodeType:: "BumpMapping"
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeBumpMapping(VrmlScene *);
    virtual ~VrmlNodeBumpMapping();

    // Copy the node.
    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeBumpMapping *toBumpMapping() const;

    virtual void addToScene(VrmlScene *s, const char *relativeUrl);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName,
                          const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

protected:
};
}
#endif //_VRMLNODEBumpMapping__
