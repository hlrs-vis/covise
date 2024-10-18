/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataDouble.h

#ifndef _VRMLNODEMETADATADOUBLE_
#define _VRMLNODEMETADATADOUBLE_

#include "VrmlNodeMetadata.h"
#include "VrmlMFDouble.h"

namespace vrml
{

#define VRML_NODE_METADATA_NUMERIC_DECL(typename, vrml_type)\
class VRMLEXPORT VrmlNodeMetadata##typename : public VrmlNodeMetadata\
{\
\
public:\
    static void initFields(VrmlNodeMetadata##typename *node, VrmlNodeType *t);\
    static const char *name();\
\
    VrmlNodeMetadata##typename(VrmlScene *);\
\
    VrmlNodeMetadata##typename *toMetadata##typename() const override;\
\
protected:\
    vrml_type d_value;\
};\


VRML_NODE_METADATA_NUMERIC_DECL(Boolean, VrmlMFBool)
VRML_NODE_METADATA_NUMERIC_DECL(Integer, VrmlMFInt)
VRML_NODE_METADATA_NUMERIC_DECL(Double, VrmlMFDouble)
VRML_NODE_METADATA_NUMERIC_DECL(Float, VrmlMFFloat)
VRML_NODE_METADATA_NUMERIC_DECL(String, VrmlMFString)


}
#endif // _VRMLNODEMETADATADOUBLE_
