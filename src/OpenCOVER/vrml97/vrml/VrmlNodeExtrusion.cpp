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

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeExtrusion(s); }

// Define the built in VrmlNodeType:: "Extrusion" fields

VrmlNodeType *VrmlNodeExtrusion::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Extrusion", creator);
    }

    VrmlNodeGeometry::defineType(t); // Parent class

    t->addEventIn("set_crossSection", VrmlField::MFVEC2F);
    t->addEventIn("set_orientation", VrmlField::MFROTATION);
    t->addEventIn("set_scale", VrmlField::MFVEC2F);
    t->addEventIn("set_spine", VrmlField::MFVEC3F);

    t->addField("beginCap", VrmlField::SFBOOL);
    t->addField("ccw", VrmlField::SFBOOL);
    t->addField("convex", VrmlField::SFBOOL);
    t->addField("creaseAngle", VrmlField::SFFLOAT);
    t->addField("crossSection", VrmlField::MFVEC2F);
    t->addField("endCap", VrmlField::SFBOOL);
    t->addField("orientation", VrmlField::MFROTATION);
    t->addField("scale", VrmlField::MFVEC2F);
    t->addField("solid", VrmlField::SFBOOL);
    t->addField("spine", VrmlField::MFVEC3F);

    return t;
}

VrmlNodeType *VrmlNodeExtrusion::nodeType() const { return defineType(0); }

VrmlNodeExtrusion::VrmlNodeExtrusion(VrmlScene *scene)
    : VrmlNodeGeometry(scene)
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

VrmlNodeExtrusion::~VrmlNodeExtrusion()
{
}

VrmlNode *VrmlNodeExtrusion::cloneMe() const
{
    return new VrmlNodeExtrusion(*this);
}

std::ostream &VrmlNodeExtrusion::printFields(std::ostream &os, int indent)
{
    if (!d_beginCap.get())
        PRINT_FIELD(beginCap);
    if (!d_endCap.get())
        PRINT_FIELD(endCap);
    if (!d_ccw.get())
        PRINT_FIELD(ccw);
    if (!d_convex.get())
        PRINT_FIELD(convex);
    if (!d_solid.get())
        PRINT_FIELD(solid);

    if (d_creaseAngle.get() != 0.0)
        PRINT_FIELD(creaseAngle);
    PRINT_FIELD(crossSection);
    PRINT_FIELD(orientation);
    PRINT_FIELD(scale);
    PRINT_FIELD(spine);

    return os;
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

// Set the value of one of the node fields.

void VrmlNodeExtrusion::setField(const char *fieldName,
                                 const VrmlField &fieldValue)
{
    if
        TRY_FIELD(beginCap, SFBool)
    else if
        TRY_FIELD(ccw, SFBool)
    else if
        TRY_FIELD(convex, SFBool)
    else if
        TRY_FIELD(creaseAngle, SFFloat)
    else if
        TRY_FIELD(crossSection, MFVec2f)
    else if
        TRY_FIELD(endCap, SFBool)
    else if
        TRY_FIELD(orientation, MFRotation)
    else if
        TRY_FIELD(scale, MFVec2f)
    else if
        TRY_FIELD(solid, SFBool)
    else if
        TRY_FIELD(spine, MFVec3f)
    else
        VrmlNodeGeometry::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeExtrusion::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "beginCap") == 0)
        return &d_beginCap;
    else if (strcmp(fieldName, "ccw") == 0)
        return &d_ccw;
    else if (strcmp(fieldName, "convex") == 0)
        return &d_convex;
    else if (strcmp(fieldName, "creaseAngle") == 0)
        return &d_creaseAngle;
    else if (strcmp(fieldName, "crossSection") == 0)
        return &d_crossSection;
    else if (strcmp(fieldName, "endCap") == 0)
        return &d_endCap;
    else if (strcmp(fieldName, "orientation") == 0)
        return &d_orientation;
    else if (strcmp(fieldName, "scale") == 0)
        return &d_scale;
    else if (strcmp(fieldName, "solid") == 0)
        return &d_solid;
    else if (strcmp(fieldName, "spine") == 0)
        return &d_spine;

    return VrmlNodeGeometry::getField(fieldName);
}

VrmlNodeExtrusion *VrmlNodeExtrusion::toExtrusion() const
{
    return (VrmlNodeExtrusion *)this;
}
