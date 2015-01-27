/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SimCudaAllocator_h__
#define __SimCudaAllocator_h__

#include <map>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace SimLib
{
class SimCudaAllocator
{
private:
    size_t mAllocatedGPU;

    typedef std::map<void *, size_t> AllocMapType;
    AllocMapType mAllocMapGPU;

public:
    SimCudaAllocator()
        : mAllocatedGPU(0)
    {
    }

    size_t GetAllocedAmount()
    {
        return mAllocatedGPU;
        // 			size_t s = 0;
        // 			for(AllocMapType::const_iterator it = mAllocMapGPU.begin(); it != mAllocMapGPU.end(); ++it)
        // 			{
        // 				s += it->second;
        // 			}
    }

    cudaError_t Allocate(void **devPtr, size_t size)
    {
        cudaError_t err = cudaMalloc(devPtr, size);
        if (err == cudaSuccess)
        {
            mAllocMapGPU[*devPtr] = size;
            mAllocatedGPU += size;
        }
        return err;
    }

    cudaError_t Free(void **devPtr)
    {
        cudaError_t err = cudaFree(*devPtr);
        if (err == cudaSuccess)
        {
            mAllocatedGPU -= mAllocMapGPU[*devPtr];
            mAllocMapGPU[devPtr] = 0;
        }
        devPtr = NULL;
        return err;
    }

    cudaError_t AllocateHost(void **devPtr, size_t size)
    {
        *devPtr = malloc(size);
        cudaError_t err = devPtr != NULL ? cudaSuccess : cudaErrorMemoryAllocation;
        return err;
    }

    cudaError_t FreeHost(void **devPtr)
    {
        free(*devPtr);
        devPtr = NULL;
        return cudaSuccess;
    }

    cudaError_t AllocateHostPinned(void **devPtr, size_t size)
    {
        cudaError_t err = cudaMallocHost(devPtr, size);
        return err;
    }

    cudaError_t FreeHostPinned(void **devPtr)
    {
        cudaError_t err = cudaFreeHost(*devPtr);
        if (err == cudaSuccess)
        {
            devPtr = NULL;
        }
        return err;
    }
};
}
#endif
