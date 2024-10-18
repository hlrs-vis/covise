/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeILineSet.cpp

#include "VrmlNodeILineSet.h"

#include "VrmlNodeType.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeTextureCoordinate.h"

#include "Viewer.h"

using namespace vrml;

void VrmlNodeILineSet::initFields(VrmlNodeILineSet *node, VrmlNodeType *t)
{
    VrmlNodeIndexedSet::initFields(node, t); // Parent class
}

const char *VrmlNodeILineSet::name() { return "IndexedLineSet"; }


VrmlNodeILineSet::VrmlNodeILineSet(VrmlScene *scene)
    : VrmlNodeIndexedSet(scene, name())
{
}

void VrmlNodeILineSet::cloneChildren(VrmlNamespace *ns)
{
    if (d_color.get())
    {
        d_color.set(d_color.get()->clone(ns));
        d_color.get()->parentList.push_back(this);
    }
    if (d_coord.get())
    {
        d_coord.set(d_coord.get()->clone(ns));
        d_coord.get()->parentList.push_back(this);
    }
}

// TO DO colors

Viewer::Object VrmlNodeILineSet::insertGeometry(Viewer *viewer)
{
    Viewer::Object obj = 0;
    if (d_coord.get())
    {
        VrmlMFVec3f &coord = d_coord.get()->toCoordinate()->coordinate();
        int nvert = coord.size();
        int ncoord = nvert;
        float *color = NULL;
        int nci = 0, *ci = NULL;
        int *localci = NULL;
        int *cdi = NULL, *localcdi = NULL;

        if (d_coordIndex.size() > 0)
        {
            cdi = &d_coordIndex[0];
            ncoord = d_coordIndex.size();
        }
        else
        {
            localcdi = new int[ncoord];
            cdi = localcdi;
            for (int i = 0; i < ncoord; i++)
            {
                cdi[i] = i;
            }
        }

        // check #colors is consistent with colorPerVtx, colorIndex...
        int componentsPerColor = 3;
        int cSize = -1;
        VrmlNode *colorNode = d_color.get();
        if (colorNode && (strcmp(colorNode->nodeType()->getName(), "ColorRGBA") == 0))
        {
            VrmlMFColorRGBA &c = d_color.get()->toColorRGBA()->color();
            color = &c[0][0];
            cSize = c.size();
            componentsPerColor = 4;
        }
        else if (d_color.get())
        {
            VrmlMFColor &c = d_color.get()->toColor()->color();
            color = &c[0][0];
            cSize = c.size();
        }
        if (cSize > -1)
        {
            nci = d_colorIndex.size();
            if (nci)
            {
                ci = d_colorIndex.get();
            }
            else
            {
                nci = cSize;
                localci = new int[cSize];
                ci = localci;
                for (int i = 0; i < nci; i++)
                {
                    ci[i] = i;
                }
            }
        }

        obj = viewer->insertLineSet(nvert, &coord[0][0],
                                    ncoord, cdi,
                                    d_colorPerVertex.get(),
                                    color, componentsPerColor,
                                    nci, ci, name());

        delete[] localcdi;
        delete[] localci;
    }

    if (d_color.get())
        d_color.get()->clearModified();
    if (d_coord.get())
        d_coord.get()->clearModified();

    return obj;
}
