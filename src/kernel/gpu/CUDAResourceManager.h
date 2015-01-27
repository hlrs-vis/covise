/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CUDARESOURCEMANAGER_H
#define CUDARESOURCEMANAGER_H

#include "GPUResourceManager.h"

/**
 * per process singleton manager of CUDA resources
 */
class GPUEXPORT CUDAResourceManager : public GPUResourceManager
{
public:
    static CUDAResourceManager *getInstance();
    virtual ~CUDAResourceManager();

    /*
   virtual GPUUsg *replaceUSG(GPUUsg *usg,
                              const int numElem, const int numConn,
                              const int numCoord, const int *typeList,
                              const int *elemList, const int *connList,
                              const float *x, const float *y, const float *z,
                              const int numElemM = 0, const int numConnM = 0,
                              const int numCoordM = 0);
*/
    virtual GPUUsg *allocUSG(const char *name,
                             const int numElem, const int numConn,
                             const int numCoord, const int *typeList,
                             const int *elemList, const int *connList,
                             const float *x, const float *y, const float *z,
                             const int numElemM = 0, const int numConnM = 0,
                             const int numCoordM = 0);

    virtual GPUScalar *allocScalar(const char *name, const int numElem,
                                   const float *data, const int numElemM = 0);

    virtual GPUVector *allocVector(const char *name, const int numElem,
                                   const float *u, const float *v, const float *w, const int numElemM = 0);

    virtual void deallocUSG(GPUUsg *usg);
    virtual void deallocScalar(GPUScalar *scalar);
    virtual void deallocVector(GPUVector *vector);

private:
    CUDAResourceManager();
    static CUDAResourceManager *instance;
};

#endif
