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

void VrmlNodeIPolygonsCommon::initFields(VrmlNodeIPolygonsCommon *node, VrmlNodeType *t)
{
    VrmlNodePolygonsCommon::initFields(node, t);
    initFieldsHelper(node, t,
                        field("index", node->d_index));
    if(t)
        t->addEventIn("set_index", VrmlField::MFINT32);

}

VrmlNodeIPolygonsCommon::VrmlNodeIPolygonsCommon(VrmlScene *scene, const std::string &name)
    : VrmlNodePolygonsCommon(scene, name)
{
}
