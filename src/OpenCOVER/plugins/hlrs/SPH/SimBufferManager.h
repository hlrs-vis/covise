/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __SimMemoryManager_h__
#define __SimMemoryManager_h__

#include "SimBuffer.h"
#include "SimCudaAllocator.h"
#include <map>

typedef unsigned int uint;

namespace SimLib
{
template <class T>
class BufferManager
{
private:
    SimCudaAllocator *mSimCudaAllocator;

    typedef std::map<T, SimBuffer *> MapType;
    MapType mBuffers;

public:
    BufferManager(SimCudaAllocator *SimCudaAllocator)
        : mSimCudaAllocator(SimCudaAllocator)

    {
    }
    ~BufferManager()
    {
        FreeBuffers();
    }

    SimBuffer *operator[](T i)
    {
        return Get(i);
    }

    SimBuffer *Get(T i)
    {
        return mBuffers[i];
    }

    // 		template<typename D>
    // 		D* GetPtr(T i)
    // 		{
    // 			return mBuffers[i]->GetPtr<D>();
    // 		}

    template <class D>
    D *GetPtr(T i)
    {
        SimBuffer *b = mBuffers[i];
        assert(sizeof(D) == b->GetElementSize());
        return (D *)b->GetPtr();
    }

    void SetBuffer(T id, SimBuffer *buffer)
    {
        mBuffers[id] = buffer;
    }

    void RemoveBuffer(T id)
    {
        mBuffers.erase(mBuffers.find(id));
    }

    void AllocBuffers(size_t elements)
    {
        for (typename MapType::const_iterator it = mBuffers.begin(); it != mBuffers.end(); ++it)
        {
            SimBuffer *b = it->second;
            b->AllocElements(elements);
        }
    }

    void FreeBuffers()
    {
        for (typename MapType::const_iterator it = mBuffers.begin(); it != mBuffers.end(); ++it)
        {
            SimBuffer *b = it->second;
            b->Free();
        }
    }

    void MemsetBuffers(int val)
    {
        for (typename MapType::const_iterator it = mBuffers.begin(); it != mBuffers.end(); ++it)
        {
            SimBuffer *b = it->second;
            b->Memset(val);
        }
    }
};
}

#endif