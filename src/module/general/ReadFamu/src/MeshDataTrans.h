/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshDataTrans.h
 * a container for mesh file data where every timestep has its own mesh.
 */

//#include "MeshDataTrans.h"  // a container for mesh file data where every timestep has its own mesh.

#ifndef __meshfilesetdata_h__
#define __meshfilesetdata_h__

#include "MeshDataStat.h" // a container for stationary mesh data.

/**
 * a container for mesh file data where every timestep has its own mesh.
 */
class MeshDataTrans
{
public:
    MeshDataTrans(){};
    virtual ~MeshDataTrans(){};

    virtual void addMesh(MeshDataStat *m) = 0;

    virtual int getNoOfMeshes(void) const = 0;

    virtual MeshDataStat *getMeshDataStat(int timeStepNo) = 0;

    virtual int getMaxNodeNo(int timeStepNo) const = 0;
};

#endif
