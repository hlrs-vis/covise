/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTextureCoordinateGenerator.h

#ifndef _VRMLNODETEXTURECOORDINATEGENERATOR_
#define _VRMLNODETEXTURECOORDINATEGENERATOR_

#include "VrmlNode.h"
#include "VrmlSFString.h"
#include "VrmlMFFloat.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeTextureCoordinateGenerator : public VrmlNode
{

public:
    // Define the fields of TextureCoordinate nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeTextureCoordinateGenerator(VrmlScene *);
    virtual ~VrmlNodeTextureCoordinateGenerator();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeTextureCoordinateGenerator *toTextureCoordinateGenerator() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

private:
    VrmlSFString d_mode;
    VrmlMFFloat d_parameter;
};
}
#endif //_VRMLNODETEXTURECOORDINATEGENERATOR_
