/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SimBufferCuda_h__
#define __SimBufferCuda_h__

#include "SimBuffer.h"
#include "SimCudaAllocator.h"

namespace SimLib
{
class SimBufferCuda : public SimBuffer
{
public:
    SimBufferCuda(SimCudaAllocator *mSimCudaAllocator, BufferLocation bufferLocation, size_t elementSize);
    ~SimBufferCuda();

    void MapBuffer();
    void UnmapBuffer();

    void Free();
    void Alloc(size_t size);
    void Memset(int val);

    size_t GetSize();

private:
    SimCudaAllocator *mSimCudaAllocator;
};
}

#endif