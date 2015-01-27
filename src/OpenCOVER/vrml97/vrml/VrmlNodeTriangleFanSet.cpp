/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTriangleFanSet.cpp

#include <cfloat>

#include "VrmlNodeTriangleFanSet.h"

#include "VrmlNodeType.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeNormal.h"

#include "MathUtils.h"

#include "Viewer.h"
using namespace vrml;

static VrmlNode *creator(VrmlScene *s) { return new VrmlNodeTriangleFanSet(s); }

// Define the built in VrmlNodeType:: "TriangleFanSet" fields

VrmlNodeType *VrmlNodeTriangleFanSet::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("TriangleFanSet", creator);
    }

    VrmlNodePolygonsCommon::defineType(t); // Parent class

    t->addExposedField("fanCount", VrmlField::MFINT32);

    return t;
}

VrmlNodeType *VrmlNodeTriangleFanSet::nodeType() const { return defineType(0); }

VrmlNodeTriangleFanSet::VrmlNodeTriangleFanSet(VrmlScene *scene)
    : VrmlNodePolygonsCommon(scene)
{
}

VrmlNodeTriangleFanSet::~VrmlNodeTriangleFanSet()
{
}

VrmlNode *VrmlNodeTriangleFanSet::cloneMe() const
{
    return new VrmlNodeTriangleFanSet(*this);
}

std::ostream &VrmlNodeTriangleFanSet::printFields(std::ostream &os, int indent)
{
    if (!d_fanCount.get())
        PRINT_FIELD(fanCount);

    VrmlNodePolygonsCommon::printFields(os, indent);

    return os;
}

Viewer::Object VrmlNodeTriangleFanSet::insertGeometry(Viewer *viewer)
{
    //cerr << "VrmlNodeTriangleFanSet::insertGeometry" << endl;

    Viewer::Object obj = 0;

    if (d_coord.get() && (d_fanCount.size() > 0))
    {
        int numFaces = 0;
        for (int i = 0; i < d_fanCount.size(); i++)
            numFaces += d_fanCount.get()[i] - 2;
        if (numFaces < 1)
            return obj;
        int coordIndexSize = numFaces * 4;
        int *coordIndexData = new int[coordIndexSize];
        int vertexCount = 0;
        int indexCount = 0;
        for (int i = 0; i < d_fanCount.size(); i++)
        {
            for (int j = 0; j < (d_fanCount.get()[i] - 2); j++)
            {
                coordIndexData[(indexCount + j) * 4] = vertexCount;
                coordIndexData[(indexCount + j) * 4 + 1] = vertexCount + j + 1;
                coordIndexData[(indexCount + j) * 4 + 2] = vertexCount + j + 2;
                coordIndexData[(indexCount + j) * 4 + 3] = -1;
            }
            indexCount += d_fanCount.get()[i] - 2;
            vertexCount += d_fanCount.get()[i];
        }

        unsigned int optMask = 0;

        if (d_ccw.get())
            optMask |= Viewer::MASK_CCW;
        if (d_solid.get())
            optMask |= Viewer::MASK_SOLID;
        if (!d_colorPerVertex.get())
        {
            std::cerr << "Warning: According to chapter 11.4.12 TriangleFanSet of "
                      << "ISO/IEC 19775-1:2008, the value of the "
                      << "colorPerVertex field is ignored and always treated "
                      << "as TRUE" << std::endl;
        }
        optMask |= Viewer::MASK_COLOR_PER_VERTEX;
        if (d_normalPerVertex.get())
            optMask |= Viewer::MASK_NORMAL_PER_VERTEX;

        VrmlMFInt coordIndex(coordIndexSize, coordIndexData);
        VrmlSFFloat creaseAngle(CREASE_ANGLE);

        obj = VrmlNodeColoredSet::insertGeometry(viewer, optMask, coordIndex,
                                                 coordIndex, creaseAngle,
                                                 d_normal, coordIndex,
                                                 d_texCoord, coordIndex);
    }

    if (d_color.get())
        d_color.get()->clearModified();
    if (d_coord.get())
        d_coord.get()->clearModified();
    if (d_normal.get())
        d_normal.get()->clearModified();
    if (d_texCoord.get())
        d_texCoord.get()->clearModified();

    return obj;
}

// Set the value of one of the node fields.

void VrmlNodeTriangleFanSet::setField(const char *fieldName,
                                      const VrmlField &fieldValue)
{
    if
        TRY_FIELD(fanCount, MFInt)
    else
        VrmlNodePolygonsCommon::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeTriangleFanSet::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "fanCount") == 0)
        return &d_fanCount;
    return VrmlNodePolygonsCommon::getField(fieldName);
}

VrmlNodeTriangleFanSet *VrmlNodeTriangleFanSet::toTriangleFanSet() const
{
    return (VrmlNodeTriangleFanSet *)this;
}
