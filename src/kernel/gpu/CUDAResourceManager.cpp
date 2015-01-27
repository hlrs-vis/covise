/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#undef _GLIBCXX_ATOMIC_BUILTINS_4

#include "CUDAResourceManager.h"
#ifdef WIN32
typedef unsigned int uint;
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

GPUEXPORT CUDAResourceManager *CUDAResourceManager::instance = NULL;
// copied from NVIDIA CUDA SDK's cutil.h

#define CUDA_SAFE_CALL(call)                                                                                            \
    {                                                                                                                   \
        cudaError err = call;                                                                                           \
        if (cudaSuccess != err)                                                                                         \
        {                                                                                                               \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                         \
        }                                                                                                               \
    }

CUDAResourceManager::CUDAResourceManager()
    : GPUResourceManager()
{
}

CUDAResourceManager *CUDAResourceManager::getInstance()
{
    if (!instance)
    {
        instance = new CUDAResourceManager();
    }

    return instance;
}
/*
GPUUsg *CUDAResourceManager::replaceUSG(GPUUsg *usg,
                                        const int numElem, const int numConn,
                                        const int numCoord, const int *typeList,
                                        const int *elemList, const int *connList,
                                        const float *x, const float *y, const float *z)
{
   cudaMemcpy(usg->getElemList(), elemList, sizeof(uint) * numElem,
              cudaMemcpyHostToDevice);
   
   cudaMemcpy(usg->getTypeList(), typeList, sizeof(uint) * numElem,
              cudaMemcpyHostToDevice);
   
   cudaMemcpy(usg->getConnList(), connList, numConn * sizeof(int),
              cudaMemcpyHostToDevice);
   
   float *vertices = usg->getVertices();
   cudaMemcpy(vertices,                x, numCoord * sizeof(float),
              cudaMemcpyHostToDevice);
   cudaMemcpy(vertices + numCoord,     y, numCoord * sizeof(float),
              cudaMemcpyHostToDevice);
   cudaMemcpy(vertices + 2 * numCoord, z, numCoord * sizeof(float),
              cudaMemcpyHostToDevice);

   return usg;
}
*/

GPUUsg *CUDAResourceManager::allocUSG(const char *name,
                                      const int numElem, const int numConn,
                                      const int numCoord, const int *typeList,
                                      const int *elemList, const int *connList,
                                      const float *x, const float *y, const float *z,
                                      const int numElemM, const int numConnM,
                                      const int numCoordM)
{
    float *g_vertices;
    int *g_typeList, *g_elemList, *g_connList;

    if (numElemM)
    {
        cudaMalloc((void **)&g_elemList, sizeof(uint) * numElemM);
        cudaMalloc((void **)&g_typeList, sizeof(uint) * numElemM);
    }
    else
    {
        cudaMalloc((void **)&g_elemList, sizeof(uint) * numElem);
        cudaMalloc((void **)&g_typeList, sizeof(uint) * numElem);
    }

    if (numConnM)
        cudaMalloc((void **)&g_connList, numConnM * sizeof(int));
    else
        cudaMalloc((void **)&g_connList, numConn * sizeof(int));

    if (numCoordM)
        cudaMalloc((void **)&g_vertices, numCoordM * 3 * sizeof(float));
    else
        cudaMalloc((void **)&g_vertices, numCoord * 3 * sizeof(float));

    cudaMemcpy(g_elemList, elemList, sizeof(uint) * numElem,
               cudaMemcpyHostToDevice);

    cudaMemcpy(g_typeList, typeList, sizeof(uint) * numElem,
               cudaMemcpyHostToDevice);

    cudaMemcpy(g_connList, connList, numConn * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(g_vertices, x, numCoord * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_vertices + numCoord, y, numCoord * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(g_vertices + 2 * numCoord, z, numCoord * sizeof(float),
               cudaMemcpyHostToDevice);

    return new GPUUsg(name, numElem, numConn, numCoord,
                      g_elemList, g_typeList, g_connList, g_vertices,
                      numElemM, numConnM, numCoordM);
}

GPUScalar *CUDAResourceManager::allocScalar(const char *name,
                                            const int numElem,
                                            const float *data,
                                            const int numElemM)
{
    float *g_data;
    if (numElemM)
        cudaMalloc((void **)&g_data, numElemM * sizeof(float));
    else
        cudaMalloc((void **)&g_data, numElem * sizeof(float));

    cudaMemcpy(g_data, data, numElem * sizeof(float), cudaMemcpyHostToDevice);

    return new GPUScalar(name, numElem, g_data, numElemM);
}

GPUVector *CUDAResourceManager::allocVector(const char *name,
                                            const int numElem,
                                            const float *u,
                                            const float *v,
                                            const float *w,
                                            const int numElemM)
{
    float *g_data;
    if (numElemM)
        cudaMalloc((void **)&g_data, numElemM * sizeof(float) * 3);
    else
        cudaMalloc((void **)&g_data, numElem * sizeof(float) * 3);

    cudaMemcpy(g_data, u, numElem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_data + numElem, v, numElem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_data + numElem * 2, w, numElem * sizeof(float), cudaMemcpyHostToDevice);

    return new GPUVector(name, numElem, g_data, numElemM);
}

void CUDAResourceManager::deallocUSG(GPUUsg *usg)
{
    CUDA_SAFE_CALL(cudaFree(usg->getElemList()));
    CUDA_SAFE_CALL(cudaFree(usg->getTypeList()));
    CUDA_SAFE_CALL(cudaFree(usg->getConnList()));
    CUDA_SAFE_CALL(cudaFree(usg->getVertices()));
}

void CUDAResourceManager::deallocScalar(GPUScalar *scalar)
{
    CUDA_SAFE_CALL(cudaFree(scalar->getData()));
}

void CUDAResourceManager::deallocVector(GPUVector *vector)
{
    CUDA_SAFE_CALL(cudaFree(vector->getData()));
}

CUDAResourceManager::~CUDAResourceManager()
{
}
