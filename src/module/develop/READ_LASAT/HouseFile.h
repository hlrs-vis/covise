/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS   Triangulator
//
// Description: This class provides a triangulation algorithm
//
//
// Initial version: 11.12.2002 (CS)
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// All Rights Reserved.
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
// $Id: HouseFile.h,v 1.3 2002/12/17 13:36:05 ralf Exp $
//
#ifndef _HOUSEFILE_H_
#define _HOUSEFILE_H_

//#include "coDistributedObject.h"
//#include <iostream>
#include <stdio.h>
#include <assert.h>

typedef int (*triang)[3];
class coDistributedObject;
class HouseFile
{

public:
    // Member functions

    HouseFile(const char *filename, float zbase, const char *o_name);
    ~HouseFile()
    {
        delete[] obj_name;
    };

    bool isValid()
    {
        return (num_nodes > 0);
    }

    coDistributedObject *getPolygon();

private:
    enum
    {
        MAXLINE = 1280
    };

    int num_nodes; // Number of nodes
    int num_polygons; // Number of elements
    int num_components; // Number of components

    float zbase_; // origin for z coordinates

    int pl[40000];
    int cl[200000];
    float xCoords[10000], yCoords[10000], zCoords[10000];

    char *obj_name;

    void addBox(int cx, int cy, int cz, int length, int width,
                int height, int angle);
    void addPrism_(int numVert, float *x, float *y, float height);
    void addPrism(int numVert, float *x, float *y, float height);
};
#endif

//
// History:
//
// $Log: HouseFile.h,v $
// Revision 1.3  2002/12/17 13:36:05  ralf
// adapted for windows
//
// Revision 1.2  2002/12/12 16:56:54  ralfm_te
// -
//
// Revision 1.1  2002/12/12 11:59:24  cs_te
// initial version
//
//
