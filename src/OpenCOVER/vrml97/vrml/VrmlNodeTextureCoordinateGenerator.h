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

#include "VrmlNodeTemplate.h"
#include "VrmlSFString.h"
#include "VrmlMFFloat.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeTextureCoordinateGenerator : public VrmlNodeTemplate
{

public:
    // Define the fields of TextureCoordinate nodes
    static void initFields(VrmlNodeTextureCoordinateGenerator *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeTextureCoordinateGenerator(VrmlScene *);

    VrmlNodeTextureCoordinateGenerator *toTextureCoordinateGenerator() const override;

private:
    VrmlSFString d_mode;
    VrmlMFFloat d_parameter;
};
}
#endif //_VRMLNODETEXTURECOORDINATEGENERATOR_
