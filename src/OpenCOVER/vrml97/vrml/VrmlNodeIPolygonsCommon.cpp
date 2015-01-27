/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeIPolygonsCommon.cpp

#include <cfloat>

#include "VrmlNodeIPolygonsCommon.h"

#include "VrmlNodeType.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeNormal.h"

#include "MathUtils.h"

#include "Viewer.h"
using namespace vrml;

// Define the built in VrmlNodeType:: "IndexedPolygons*Set" fields

VrmlNodeType *VrmlNodeIPolygonsCommon::defineType(VrmlNodeType *t)
{
    if (!t)
    {
        return NULL;
    }
    VrmlNodePolygonsCommon::defineType(t); // Parent class

    t->addEventIn("set_index", VrmlField::MFINT32);
    t->addField("index", VrmlField::MFINT32);

    return t;
}

VrmlNodeIPolygonsCommon::VrmlNodeIPolygonsCommon(VrmlScene *scene)
    : VrmlNodePolygonsCommon(scene)
{
}

VrmlNodeIPolygonsCommon::~VrmlNodeIPolygonsCommon()
{
}

std::ostream &VrmlNodeIPolygonsCommon::printFields(std::ostream &os, int indent)
{
    if (!d_index.get())
        PRINT_FIELD(index);

    VrmlNodePolygonsCommon::printFields(os, indent);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeIPolygonsCommon::setField(const char *fieldName,
                                       const VrmlField &fieldValue)
{
    if
        TRY_FIELD(index, MFInt)
    else
        VrmlNodePolygonsCommon::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeIPolygonsCommon::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "index") == 0)
        return &d_index;

    return VrmlNodePolygonsCommon::getField(fieldName);
}
