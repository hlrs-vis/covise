/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __osgSimBuffer_h__
#define __osgSimBuffer_h__

#include "SimBuffer.h"
#include "SPH.h"

class osgSimBuffer : public SimLib::SimBuffer
{
public:
    osgSimBuffer(fluidDrawable *_fd);
    ~osgSimBuffer();

    void SetVBO(unsigned int vbo);

    virtual void MapBuffer();
    virtual void UnmapBuffer();

    virtual void Alloc(size_t size);
    virtual void Memset(int val);
    virtual void Free();
    virtual size_t GetSize();
    unsigned int getVBO()
    {
        return mVBO;
    };

private:
    unsigned int mVBO;

    struct cudaGraphicsResource *cuda_resource; // handles OpenGL-CUDA exchange
    fluidDrawable *drawable;
};
#endif
