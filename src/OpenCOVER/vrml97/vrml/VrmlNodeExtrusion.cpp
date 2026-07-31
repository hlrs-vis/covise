/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeExtrusion.cpp

#include "VrmlNodeExtrusion.h"

#include "VrmlNodeType.h"
#include "Viewer.h"

using namespace vrml;

void VrmlNodeExtrusion::initFields(VrmlNodeExtrusion *node, VrmlNodeType *t)
{
    VrmlNodeGeometry::initFields(node, t); // Parent class

    initFieldsHelper(node, t,
                     field("beginCap", &VrmlNodeExtrusion::d_beginCap),
                     field("ccw", &VrmlNodeExtrusion::d_ccw),
                     field("convex", &VrmlNodeExtrusion::d_convex),
                     field("creaseAngle", &VrmlNodeExtrusion::d_creaseAngle),
                     field("crossSection", &VrmlNodeExtrusion::d_crossSection),
                     field("endCap", &VrmlNodeExtrusion::d_endCap),
                     field("orientation", &VrmlNodeExtrusion::d_orientation),
                     field("scale", &VrmlNodeExtrusion::d_scale),
                     field("solid", &VrmlNodeExtrusion::d_solid),
                     field("spine", &VrmlNodeExtrusion::d_spine));
    if(t)
    {   
        t->addEventIn("set_crossSection", VrmlField::MFVEC2F);
        t->addEventIn("set_orientation", VrmlField::MFROTATION);
        t->addEventIn("set_scale", VrmlField::MFVEC2F);
        t->addEventIn("set_spine", VrmlField::MFVEC3F);}
}

const char *VrmlNodeExtrusion::typeName() { return "Extrusion"; }


VrmlNodeExtrusion::VrmlNodeExtrusion(VrmlScene *scene)
    : VrmlNodeGeometry(scene, typeName())
    , d_beginCap(true)
    , d_ccw(true)
    , d_convex(true)
    , d_endCap(true)
    , d_orientation(0, 0, 1, 0)
    , d_scale(1, 1)
    , d_solid(true)
{
    float crossSection[] = { 1, 1, 1, -1, -1, -1, -1, 1, 1, 1 };
    d_crossSection.set(5, crossSection);
    float spine[] = { 0, 0, 0, 0, 1, 0 };
    d_spine.set(2, spine);
}

Viewer::Object VrmlNodeExtrusion::insertGeometry(Viewer *viewer)
{
    Viewer::Object obj = 0;
    if (d_crossSection.size() > 0 && d_spine.size() > 1)
    {
        unsigned int optMask = 0;
        if (d_ccw.get())
            optMask |= Viewer::MASK_CCW;
        if (d_convex.get())
            optMask |= Viewer::MASK_CONVEX;
        if (d_solid.get())
            optMask |= Viewer::MASK_SOLID;
        if (d_beginCap.get())
            optMask |= Viewer::MASK_BOTTOM;
        if (d_endCap.get())
            optMask |= Viewer::MASK_TOP;

        obj = viewer->insertExtrusion(optMask,
                                      d_orientation.size(),
                                      d_orientation.get(),
                                      d_scale.size(),
                                      d_scale.get(),
                                      d_crossSection.size(),
                                      d_crossSection.get(),
                                      d_spine.size(),
                                      d_spine.get(),
                                      d_creaseAngle.get());
    }

    return obj;
}
