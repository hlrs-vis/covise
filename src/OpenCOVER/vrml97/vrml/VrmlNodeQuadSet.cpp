/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeQuadSet.cpp

#include <cfloat>

#include "VrmlNodeQuadSet.h"

#include "VrmlNodeType.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeNormal.h"

#include "MathUtils.h"

#include "Viewer.h"
using namespace vrml;

void VrmlNodeQuadSet::initFields(VrmlNodeQuadSet *node, VrmlNodeType *t)
{
    VrmlNodePolygonsCommon::initFields(node, t); 
}

const char *VrmlNodeQuadSet::name() { return "QuadSet"; }


VrmlNodeQuadSet::VrmlNodeQuadSet(VrmlScene *scene)
    : VrmlNodePolygonsCommon(scene, name())
{
}

Viewer::Object VrmlNodeQuadSet::insertGeometry(Viewer *viewer)
{
    //cerr << "VrmlNodeQuadSet::insertGeometry" << endl;

    Viewer::Object obj = 0;

    if (d_coord.get())
    {
        int numPoints = d_coord.get()->getNumberCoordinates();
        int coordIndexSize = numPoints * 5;
        int *coordIndexData = new int[coordIndexSize];

        for (int i = 0; i < numPoints; i++)
        {
            for (int j = 0; j < 4; j++)
                coordIndexData[i * 5 + j] = i * 4 + j;
            coordIndexData[i * 5 + 4] = -1;
        }

        unsigned int optMask = 0;

        if (d_ccw.get())
            optMask |= Viewer::MASK_CCW;
        if (d_solid.get())
            optMask |= Viewer::MASK_SOLID;
        if (!d_colorPerVertex.get())
        {
            std::cerr << "Warning: According to chapter 32.4.6 QuadSet of "
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
