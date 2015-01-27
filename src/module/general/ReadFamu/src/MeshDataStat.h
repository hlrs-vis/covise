/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshDataTrans.h
 * a container for stationary mesh data.
 */

//#include "MeshDataStat.h"  // a container for stationary mesh data.

#ifndef __MeshDataStat_h__
#define __MeshDataStat_h__

/**
 * a container for stationary mesh data.
 */
class MeshDataStat
{
public:
    MeshDataStat(){};
    virtual ~MeshDataStat(){};

    virtual int getNoOfElements(void) const = 0;
    virtual int getNoOfPoints(void) const = 0;

    virtual void getMeshData(int *noOfElements, int *noOfVertices, int *noOfPoints,
                             int *elementsArr[], int *verticesArr[],
                             float *xPointsArr[], float *yPointsArr[], float *zPointsArr[],
                             int *typesArr[]) = 0;

    // returns -1 if meshNodeNo is too high.
    virtual int getInternalNodeNo(int meshNodeNo) const = 0;
    virtual int getMaxNodeNo(void) const = 0;
};

#endif
