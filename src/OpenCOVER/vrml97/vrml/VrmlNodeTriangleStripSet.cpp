/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeTriangleStripSet.cpp

#include <cfloat>

#include "VrmlNodeTriangleStripSet.h"

#include "VrmlNodeType.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeNormal.h"

#include "MathUtils.h"

#include "Viewer.h"
using namespace vrml;

void VrmlNodeTriangleStripSet::initFields(VrmlNodeTriangleStripSet *node, VrmlNodeType *t)
{
    VrmlNodePolygonsCommon::initFields(node, t); // Parent class
    initFieldsHelper(node, t, exposedField("stripCount", node->d_stripCount));

}

const char *VrmlNodeTriangleStripSet::name()
{
    return "TriangleStripSet";
}

VrmlNodeTriangleStripSet::VrmlNodeTriangleStripSet(VrmlScene *scene)
    : VrmlNodePolygonsCommon(scene, name())
{
}

Viewer::Object VrmlNodeTriangleStripSet::insertGeometry(Viewer *viewer)
{
    //cerr << "VrmlNodeTriangleStripSet::insertGeometry" << endl;

    Viewer::Object obj = 0;

    if (d_coord.get() && (d_stripCount.size() > 0))
    {
        int numFaces = 0;
        for (int i = 0; i < d_stripCount.size(); i++)
            numFaces += d_stripCount.get()[i] - 2;
        if (numFaces < 1)
            return obj;
        int coordIndexSize = numFaces * 4;
        int *coordIndexData = new int[coordIndexSize];
        int vertexCount = 0;
        int indexCount = 0;
        for (int i = 0; i < d_stripCount.size(); i++)
        {
            bool flip = false;
            for (int j = 0; j < (d_stripCount.get()[i] - 2); j++)
            {
                coordIndexData[(indexCount + j) * 4] = vertexCount + j;
                int j1 = j + 1;
                if (flip)
                    j1++;
                coordIndexData[(indexCount + j) * 4 + 1] = vertexCount + j1;
                int j2 = j + 2;
                if (flip)
                    j2--;
                coordIndexData[(indexCount + j) * 4 + 2] = vertexCount + j2;
                coordIndexData[(indexCount + j) * 4 + 3] = -1;
                flip = !flip;
            }
            indexCount += d_stripCount.get()[i] - 2;
            vertexCount += d_stripCount.get()[i];
        }

        unsigned int optMask = 0;

        if (d_ccw.get())
            optMask |= Viewer::MASK_CCW;
        if (d_solid.get())
            optMask |= Viewer::MASK_SOLID;
        if (!d_colorPerVertex.get())
        {
            std::cerr << "Warning: According to chapter 11.4.12 TriangleStripSet of "
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

VrmlNodeTriangleStripSet *VrmlNodeTriangleStripSet::toTriangleStripSet() const
{
    return (VrmlNodeTriangleStripSet *)this;
}
