/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeITriangleFanSet.cpp

#include <cfloat>

#include "VrmlNodeITriangleFanSet.h"

#include "VrmlNodeType.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeNormal.h"

#include "MathUtils.h"

#include "Viewer.h"
using namespace vrml;

void VrmlNodeITriangleFanSet::initFields(VrmlNodeITriangleFanSet *node, VrmlNodeType *t)
{
    VrmlNodeIPolygonsCommon::initFields(node, t);
}

const char *VrmlNodeITriangleFanSet::name() { return "IndexedTriangleFanSet"; }

VrmlNodeITriangleFanSet::VrmlNodeITriangleFanSet(VrmlScene *scene)
    : VrmlNodeIPolygonsCommon(scene, name())
{
}

Viewer::Object VrmlNodeITriangleFanSet::insertGeometry(Viewer *viewer)
{
    //cerr << "VrmlNodeITriangleFanSet::insertGeometry" << endl;

    Viewer::Object obj = 0;

    if (d_coord.get() && d_index.size() > 0)
    {
        int coordIndexSize = 0;
        // account coordIndex size
        int fanSetFirst = -1;
        int fanSetSecond = -1;
        for (int i = 0; i < d_index.size(); i++)
        {
            int currentIndex = d_index.get()[i];
            if (currentIndex == -1)
            {
                fanSetFirst = -1;
                fanSetSecond = -1;
            }
            else
            {
                if (fanSetFirst == -1)
                {
                    // first vertex of fan
                    fanSetFirst = i;
                }
                else if (fanSetSecond == -1)
                {
                    // second vertex of fan
                    fanSetSecond = i;
                }
                else
                {
                    // other vertices of fan
                    coordIndexSize += 4;
                    fanSetSecond = i;
                }
            }
        }
        // fill coordIndex
        int *coordIndexData = new int[coordIndexSize];
        fanSetFirst = -1;
        fanSetSecond = -1;
        coordIndexSize = 0;
        for (int i = 0; i < d_index.size(); i++)
        {
            int currentIndex = d_index.get()[i];
            if (currentIndex == -1)
            {
                fanSetFirst = -1;
                fanSetSecond = -1;
            }
            else
            {
                if (fanSetFirst == -1)
                {
                    // first vertex of fan
                    fanSetFirst = i;
                }
                else if (fanSetSecond == -1)
                {
                    // second vertex of fan
                    fanSetSecond = i;
                }
                else
                {
                    // other vertices of fan
                    coordIndexData[coordIndexSize++] = d_index.get()[fanSetFirst];
                    coordIndexData[coordIndexSize++] = d_index.get()[fanSetSecond];
                    coordIndexData[coordIndexSize++] = currentIndex;
                    coordIndexData[coordIndexSize++] = -1;
                    fanSetSecond = i;
                }
            }
        }

        unsigned int optMask = 0;

        if (d_ccw.get())
            optMask |= Viewer::MASK_CCW;
        if (d_solid.get())
            optMask |= Viewer::MASK_SOLID;
        if (!d_colorPerVertex.get())
        {
            std::cerr << "Warning: According to chapter 11.4.6 IndexedTriangleFanSet of "
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
