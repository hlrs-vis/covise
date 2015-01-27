/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CUDA_STATE_H
#define CUDA_STATE_H

//#define CUDPP_STATIC_LIB
#include <cudpp.h>
#include <cuda_runtime.h>

#include "gpu/GPUResourceManager.h"

typedef unsigned int GLuint;
typedef unsigned int uint;
typedef unsigned char uchar;

struct CUDAState
{
    // tables (
    uint *d_hexaNumVertsTable;
    uint *d_hexaTriTable;
    uint *d_tetraNumVertsTable;
    uint *d_tetraTriTable;
    uint *d_pyrNumVertsTable;
    uint *d_pyrTriTable;
    CUDPPHandle *cudpp;

    //Threads block size for the simple kernels
    //Preferred to be a multiple of 64 (refer to the docs)
    //const int THREAD_N = 192;
    int THREAD_N;

    // Threads block size for the "fat" kernels
    // Can lead to underutilization, but needed for shared mem/ regs pressure
    int THREAD_N_FAT;

    CUDAState()
        : d_hexaNumVertsTable(0)
        , d_hexaTriTable(0)
        , d_tetraNumVertsTable(0)
        , d_tetraTriTable(0)
        , d_pyrNumVertsTable(0)
        , d_pyrTriTable(0)
        , cudpp(0)
    {
    }
};

// Unstructured Grid data on GPU
struct State
{

    uint activeElements;
    uint activeVerts;
    uint activeCoords;

    // device data
    GLuint vertexBuffer;
    GLuint normalBuffer;
    GLuint texcoordBuffer;
    GLuint licResultBuffer;
    GLuint licTexcoordBuffer;

    struct cudaGraphicsResource *vertexBufferResource;
    struct cudaGraphicsResource *normalBufferResource;
    struct cudaGraphicsResource *texcoordBufferResource;
    struct cudaGraphicsResource *licResultBufferResource;
    struct cudaGraphicsResource *licTexcoordBufferResource;

    float *d_vertexBuffer;
    float *d_normalBuffer;

    float texMin;
    float texMax;

    //bool vboCreated;
    int bufferSize;

    CUDPPHandle scanplan;

    GPUUsg *usg;
    GPUScalar *data;
    GPUScalar *mapping;
    GPUScalar *licMapping;

    size_t d_elemPitch;
    uint *d_elemClassification;
    uint *d_elemVerts;

    size_t d_scanPitch;
    uint *d_scan;
    uint *d_vertsScan;
    uint *d_compactedArray;

    GPUScalar *buffer;
    cudaStream_t stream;
    cudaStream_t cstream;
};

#endif //CUDA_STATE_H
