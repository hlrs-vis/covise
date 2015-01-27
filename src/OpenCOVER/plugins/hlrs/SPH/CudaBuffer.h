/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CudaBuffer_h__
#define __CudaBuffer_h__

namespace SimLib
{
class CudaBuffer
{
public:
    ~CudaBuffer(){};

    virtual void MapBuffer(void **devPtr) = 0;
    virtual void UnmapBuffer(void **devPtr) = 0;
    virtual void Register() = 0;
    virtual void Unregister() = 0;
    //virtual size_t GetSize()= 0;

    bool IsMapped()
    {
        return mMapped;
    }
    bool IsRegistered()
    {
        return mRegistered;
    }

protected:
    bool mMapped;
    bool mRegistered;
};
}
#endif