/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DmnaFiles_H_
#define _DmnaFiles_H_

//#include "coDoRectilinearGrid.h"
#include <api/coStepFile.h>

//#include <iostream>
//#include <stdio.h>
//#include <assert.h>
//#include <fstream>
#include <api/coModule.h>
using namespace covise;

const int MAXLINE = 1280;
class DmnaFiles
{
private:
    enum
    {
        DIM = 100
    };

    int num_nodes; // Number of nodes
    int xDim, yDim, zDim;
    float delta, yMin, xMin;
    coStepFile *files;
    float ***sdata;
    char *obj_name;

public:
    // Member functions

    DmnaFiles(const char *firstFile, const char *zFile, const char *o_name, char *&errMsg);
    ~DmnaFiles();

    bool isValid()
    {
        return (xDim > 0 && yDim > 0 && zDim > 0);
    }

    coDistributedObject *getData();
    int getXDim()
    {
        return xDim;
    }
    int getYDim()
    {
        return yDim;
    }
    float getDelta()
    {
        return delta;
    }
    float getYMin()
    {
        return yMin;
    }
    float getXMin()
    {
        return xMin;
    }
};
#endif

//
// History:
//
// $Log: DmnaFiles.h,v $
// Revision 1.2  2002/12/17 13:36:05  ralf
// adapted for windows
//
// Revision 1.1  2002/12/12 11:59:24  cs_te
// initial version
//
//
