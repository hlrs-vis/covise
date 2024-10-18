/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMetadataDouble.cpp

#include "VrmlNodeMetadataNumeric.h"

#include "VrmlNodeType.h"

#include "Viewer.h"
using namespace vrml;

#define CONCATENATE(x, y) x ## y
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)
#define STRINGIFY(x) #x

#define VRML_NODE_METADATA_NUMERIC_IMPL(typename, vrml_type) \
    void VrmlNodeMetadata##typename::initFields(VrmlNodeMetadata##typename *node, VrmlNodeType *t) \
    { \
        VrmlNodeMetadata::initFields(node, t); \
        initFieldsHelper(node, t, exposedField("value", node->d_value)); \
    } \
    const char *VrmlNodeMetadata##typename::name() { return EXPAND_AND_STRINGIFY(CONCATENATE(Metadata, typename)); } \
    VrmlNodeMetadata##typename::VrmlNodeMetadata##typename(VrmlScene *scene) \
        : VrmlNodeMetadata(scene, name()) \
    { \
    } \
    VrmlNodeMetadata##typename *VrmlNodeMetadata##typename::toMetadata##typename() const \
    { \
        return (VrmlNodeMetadata##typename *)this; \
    }


VRML_NODE_METADATA_NUMERIC_IMPL(Boolean, VrmlMFBool)
VRML_NODE_METADATA_NUMERIC_IMPL(Integer, VrmlMFInt)
VRML_NODE_METADATA_NUMERIC_IMPL(Double, VrmlMFDouble)
VRML_NODE_METADATA_NUMERIC_IMPL(Float, VrmlMFFloat)
VRML_NODE_METADATA_NUMERIC_IMPL(String, VrmlMFString)
