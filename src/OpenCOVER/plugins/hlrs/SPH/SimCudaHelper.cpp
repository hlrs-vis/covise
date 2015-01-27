/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SimCudaHelper.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <limits>

#include <cuda_runtime_api.h>

using namespace std;

namespace SimLib
{
SimCudaHelper::SimCudaHelper()
{
    mDeviceProp = new cudaDeviceProp();
}

SimCudaHelper::~SimCudaHelper()
{
}

void SimCudaHelper::Initialize(int cudaDevice)
{
    cudaDevice = Init(cudaDevice);
    cudaError res;
    // From CUDA prog guide: cudaSetDevice()and cudaGLSetGLDevice() are mutually exclusive.
    res = cudaSetDevice(cudaDevice);
    if (res != cudaSuccess)
    {
        CheckError(res, "cudaGetDeviceCount failed");
    }
    else
    {
        cout << "CUDA: Successful cudaSetDevice, using device " << cudaDevice << "\n";
    }
}

int SimCudaHelper::Init(int cudaDevice)
{
    cout << "*** Initializing CUDA system ***\n";

    cudaError res;
    int count;

    res = cudaGetDeviceCount(&count);
    CheckError("cudaGetDeviceCount failed");

    if (cudaDevice >= count)
    {
        cout << "SimCudaHelper::Initialize: Cuda device " << cudaDevice << " does not exist, trying to use " << count << " instead\n";
        cudaDevice = count;
    }

    PrintDevices(cudaDevice);

    cudaGetDeviceProperties(mDeviceProp, cudaDevice);

    // 		if(device >= count || device < 0)
    // 		{
    // 			device = cutGetMaxGflopsDeviceId();
    // 		}
    return cudaDevice;
}

#ifdef SPHSIMLIB_3D_SUPPORT

void SimCudaHelper::InitializeGL(int cudaDevice)
{
    cudaDevice = Init(cudaDevice);
    cudaError res = cudaGLSetGLDevice(cudaDevice);
    if (res != cudaSuccess)
    {
        CheckError("cudaGLSetGLDevice failed");
    }
    else
    {
        cout << "CUDA: Successful cudaGLSetGLDevice\n";
    }
}

/*	void SimCudaHelper::InitializeD3D9(int cudaDevice, IDirect3DDevice9 *pDxDevice)
	{
		cudaDevice = Init(cudaDevice);
		cudaError res = cudaD3D9SetDirect3DDevice(pDxDevice);
		if(res != cudaSuccess)
		{
			CheckError("cudaGLSetGLDevice failed");
		}
		else 
		{		
			cout << "CUDA: Successful cudaD3D9SetDirect3DDevice\n";
		}
	}*/

cudaError_t SimCudaHelper::MapBuffer(void **devPtr, GLuint bufObj)
{
    return cudaGLMapBufferObject(devPtr, bufObj);
}

cudaError_t SimCudaHelper::UnmapBuffer(void **devPtr, GLuint bufObj)
{
    cudaError err = cudaGLUnmapBufferObject(bufObj);
    *devPtr = 0;
    return err;
}

/*	cudaError_t SimCudaHelper::MapBuffer(void **devPtr, IDirect3DResource9* pResource)
	{
		cudaD3D9MapResources(1, &pResource);
		return cudaD3D9ResourceGetMappedPointer(devPtr, pResource, 0, 0);
	}	

	cudaError_t SimCudaHelper::UnmapBuffer(void **devPtr, IDirect3DResource9* pResource)
	{
		cudaError err = cudaD3D9UnmapResources(1, &pResource);
		*devPtr = 0;
		return err;
	}	

	cudaError_t SimCudaHelper::RegisterD3D9Buffer(IDirect3DResource9* pResource)
	{
		return cudaD3D9RegisterResource(pResource, cudaD3D9RegisterFlagsNone);
	}

	cudaError_t SimCudaHelper::UnregisterD3D9Buffer(IDirect3DResource9* pResource)
	{
		return cudaD3D9RegisterResource(pResource, cudaD3D9RegisterFlagsNone);
	}*/

cudaError_t SimCudaHelper::RegisterGLBuffer(GLuint vbo)
{
    return cudaGLRegisterBufferObject(vbo);
}

cudaError_t SimCudaHelper::UnregisterGLBuffer(GLuint vbo)
{
    return cudaGLUnregisterBufferObject(vbo);
}

#endif

void SimCudaHelper::CheckError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    CheckError(err, msg);
}

void SimCudaHelper::CheckError(cudaError_t err, const char *msg)
{
    if (cudaSuccess != err)
    {
        cout << "SimCudaHelper: " << msg << " " << cudaGetErrorString(err) << "\n";
    }
}

bool SimCudaHelper::IsFermi()
{
    return mDeviceProp->major == 2;
}

int SimCudaHelper::PrintDevices(int deviceSelected)
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
    {
        printf("cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
        printf("\nFAILED\n");
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
        printf("There is no device supporting CUDA\n");

    int dev;
    int driverVersion = 0, runtimeVersion = 0;
    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0)
        {
            // This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n", deviceCount);
        }
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

#if CUDART_VERSION >= 2020
        // Console log
        cudaDriverGetVersion(&driverVersion);
        printf("  CUDA Driver Version:                           %d.%d\n", driverVersion / 1000, driverVersion % 100);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion / 1000, runtimeVersion % 100);
#endif
        printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
        printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);

        printf("  Total amount of global memory:                 %zu bytes\n", deviceProp.totalGlobalMem);
#if CUDART_VERSION >= 2000
        printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
//printf("  Number of cores:                               %d\n", nGpuArchCoresPerSM[deviceProp.major] * deviceProp.multiProcessorCount);
#endif
        printf("  Total amount of constant memory:               %zu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %zu bytes\n", deviceProp.textureAlignment);
        printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
        printf("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 2020
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ? "Default (multiple host threads can use this device simultaneously)" : deviceProp.computeMode == cudaComputeModeExclusive ? "Exclusive (only one host thread at a time can use this device)" : deviceProp.computeMode == cudaComputeModeProhibited ? "Prohibited (no host thread can use this device)" : "Unknown");
#endif
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[10];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000, driverVersion % 100);
#else
    sprintf(cTemp, "%d.%d", driverVersion / 1000, driverVersion % 100);
#endif
    sProfileString += cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000, runtimeVersion % 100);
#else
    sprintf(cTemp, "%d.%d", runtimeVersion / 1000, runtimeVersion % 100);
#endif
    sProfileString += cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    sprintf(cTemp, "%d", deviceCount);
#endif
    sProfileString += cTemp;

    // First 2 device names, if any
    for (dev = 0; dev < ((deviceCount > 2) ? 2 : deviceCount); ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        sProfileString += ", Device = ";
        sProfileString += deviceProp.name;
    }
    sProfileString += "\n";
    //		shrLogEx(LOGBOTH | MASTER, 0, sProfileString.c_str());

    // finish
    //printf("\n\nPASSED\n");

    //Log* pLog = Ogre::LogManager::getSingleton().getDefaultLog();
    cudaError_t err = cudaSuccess;
    return err;
}
}
