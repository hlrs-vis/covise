/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SimCudaHelper_h__
#define __SimCudaHelper_h__

#include "Config.h"
//#define SPHSIMLIB_3D_SUPPORT

#if !defined(__CUDACC__)
#ifdef SPHSIMLIB_3D_SUPPORT

#include "GL/glew.h"
#include <cuda_gl_interop.h>

#endif
#endif

#if !defined(__CUDACC__)
#ifdef SPHSIMLIB_3D_SUPPORT
#ifdef _WIN32
//#include <d3dx9.h>

// includes, cuda
#include <cuda.h>
#include <builtin_types.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
//#include <cuda_d3d9_interop.h>
#endif
#endif
#endif

namespace SimLib
{
class SimCudaHelper
{
public:
    SimCudaHelper();
    ~SimCudaHelper();

    void Initialize(int cudaDevice);

#if !defined(__CUDACC__)
#ifdef SPHSIMLIB_3D_SUPPORT
    void InitializeGL(int cudaDevice);
    //void InitializeD3D9(int cudaDevice, IDirect3DDevice9 *pDxDevice);

    // CUDA REGISTER
    static cudaError_t RegisterGLBuffer(GLuint vbo);
    static cudaError_t UnregisterGLBuffer(GLuint vbo);

#ifdef _WIN32
//static cudaError_t RegisterD3D9Buffer(IDirect3DResource9 * pResource);
//static cudaError_t UnregisterD3D9Buffer(IDirect3DResource9 * pResource);
#endif
    // CUDA MAPPING
    //static cudaError_t MapBuffer(void **devPtr, IDirect3DResource9* pResource);
    //static cudaError_t UnmapBuffer(void **devPtr, IDirect3DResource9* pResource);

    static cudaError_t MapBuffer(void **devPtr, GLuint bufObj);
    static cudaError_t UnmapBuffer(void **devPtr, GLuint bufObj);

#endif
#endif

    int PrintDevices(int deviceSelected);

    bool IsFermi();

private:
    int Init(int cudaDevice);
    void CheckError(const char *msg);
    void CheckError(cudaError_t err, const char *msg);

    cudaDeviceProp *mDeviceProp;
};
}

#endif