/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SimBuffer_h__
#define __SimBuffer_h__

#include <assert.h>
#include <cstddef>

typedef unsigned int uint;

#include <limits.h>

#define COMPILE_TIME_ASSERT(pred) \
    switch (0)                    \
    {                             \
    case 0:                       \
    case pred:                    \
        ;                         \
    }

namespace SimLib
{
enum BufferLocation
{
    Host,
    HostPinned,
    Device
};

class SimBuffer
{
public:
    SimBuffer(BufferLocation bufferLocation, size_t elementSize)
        : mBufferLocation(bufferLocation)
        , mElementSize(elementSize)
        , mAllocedSize(0)
        , mMapped(false){};
    virtual ~SimBuffer(){};

    virtual void MapBuffer() = 0;
    virtual void UnmapBuffer() = 0;
    virtual void Alloc(size_t size) = 0;

    virtual void Memset(int val) = 0;
    virtual void Free() = 0;
    virtual size_t GetSize() = 0;

    void *GetPtr()
    {
        return mPtr;
    };

    template <typename T>
    T *GetPtr()
    {
        assert(sizeof(T) == mElementSize);
        return (T *)GetPtr();
    }

    void AllocElements(size_t elements)
    {
        Alloc(elements * mElementSize);
    }

    size_t GetElementSize()
    {
        return mElementSize;
    }

    bool IsMapped()
    {
        return mMapped;
    }

protected:
    bool mMapped;

    void *mPtr;
    size_t mSize;
    size_t mAllocedSize;
    size_t mElementSize;
    BufferLocation mBufferLocation;
};
}

#endif
