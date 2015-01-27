/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SimBufferCuda.h"

#include <memory.h>

#include <cuda_runtime_api.h>
#include "cutil.h"
#include <driver_types.h>

typedef unsigned int uint;

namespace SimLib
{
SimBufferCuda::SimBufferCuda(SimCudaAllocator *SimCudaAllocator, BufferLocation bufferLocation, size_t elementSize)
    : SimBuffer(bufferLocation, elementSize)
    , mSimCudaAllocator(SimCudaAllocator){};

SimBufferCuda::~SimBufferCuda(){};

void SimBufferCuda::MapBuffer(){};

void SimBufferCuda::UnmapBuffer(){};

void SimBufferCuda::Alloc(size_t size)
{
    mSize = size;
    if (mAllocedSize > 0)
    {
        if (mAllocedSize == mSize)
            return;

        Free();
    }

    cudaError_t result;
    switch (mBufferLocation)
    {
    case Host:
        result = mSimCudaAllocator->AllocateHost(&mPtr, mSize);
        break;
    case HostPinned:
        result = mSimCudaAllocator->AllocateHostPinned(&mPtr, mSize);
        break;
    case Device:
        result = mSimCudaAllocator->Allocate(&mPtr, mSize);
        break;
    }

    CUDA_SAFE_CALL_NO_SYNC(result)
    mAllocedSize = mSize;
}

void SimBufferCuda::Free()
{
    if (mAllocedSize > 0)
    {
        cudaError_t result;
        switch (mBufferLocation)
        {
        case Host:
            result = mSimCudaAllocator->FreeHost(&mPtr);
            break;
        case HostPinned:
            result = mSimCudaAllocator->FreeHostPinned(&mPtr);
            break;
        case Device:
            result = mSimCudaAllocator->Free(&mPtr);
            break;
        }

        CUDA_SAFE_CALL_NO_SYNC(result)
        mAllocedSize = 0;
    }
}

void SimBufferCuda::Memset(int val)
{
    if (mAllocedSize == 0)
        return;

    cudaError_t result;
    switch (mBufferLocation)
    {
    case Host:
        memset(mPtr, val, mAllocedSize);
        result = cudaSuccess;
        break;
    case HostPinned:
        memset(mPtr, val, mAllocedSize);
        result = cudaSuccess;
        break;
    case Device:
        result = cudaMemset(mPtr, val, mAllocedSize);
        break;
    }

    CUDA_SAFE_CALL_NO_SYNC(result)
}

size_t SimBufferCuda::GetSize()
{
    return mAllocedSize;
}
}
