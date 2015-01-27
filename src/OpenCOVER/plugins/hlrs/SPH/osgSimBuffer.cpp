/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "osgSimBuffer.h"

osgSimBuffer::osgSimBuffer(fluidDrawable *_fd)
    : drawable(_fd)
    , SimBuffer(SimLib::BufferLocation::Device, sizeof(float) * 4)
{
    cuda_resource = NULL;
    mVBO = 0;
    mAllocedSize = 0;
    mSize = 0;
}

osgSimBuffer::~osgSimBuffer()
{
}
/*
void osgSimBuffer::SetVBO(unsigned int vbo)
{
	bool wasMapped = false;
	if(mVBO != vbo)
	{
		if(mMapped)
		{
			wasMapped = true;
			UnmapBuffer();
		}

	}
	mVBO = vbo;
	mAllocedSize = mOgreVertexBuffer->getSizeInBytes();
	mSize = mAllocedSize;

	if(wasMapped)
		MapBuffer();

}*/

void osgSimBuffer::MapBuffer()
{
    if (cuda_resource != NULL)
    {
        mMapped = true;

        cudaGraphicsMapResources(1, &cuda_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&mPtr, &num_bytes, cuda_resource);
    }
}

void osgSimBuffer::UnmapBuffer()
{
    if (cuda_resource != NULL)
    {
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        mMapped = false;
    }
}

size_t osgSimBuffer::GetSize()
{
    return mAllocedSize;
}

void osgSimBuffer::Alloc(size_t size)
{
    if (size == mAllocedSize)
        return;

    if (!mVBO)
    {
        glGenBuffers(1, (GLuint *)&mVBO);
    }
    mAllocedSize = size;
    glBindBuffer(GL_ARRAY_BUFFER_ARB, mVBO);
    glBufferData(GL_ARRAY_BUFFER_ARB, mAllocedSize, 0, GL_STATIC_DRAW_ARB);
    glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);

    cudaError_t err = cudaSuccess;
    err = cudaGraphicsGLRegisterBuffer(&cuda_resource, mVBO, cudaGraphicsMapFlagsNone);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed (error code: %s)!\n", cudaGetErrorString(err));
    }

    err = cudaMalloc((void **)&mPtr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Cuda Malloc of %ld bytes failed (error code: %s)!\n",
                (long)size, cudaGetErrorString(err));
    }
}

void osgSimBuffer::Memset(int val)
{
    if (cuda_resource != NULL)
    {
        cudaError_t err = cudaSuccess;
        err = cudaMemset((void *)mPtr, val, mAllocedSize);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Cuda Memset of %ld bytes failed (error code: %s)!\n",
                    (long)mAllocedSize, cudaGetErrorString(err));
        }
    }
}

void osgSimBuffer::Free()
{

    glDeleteBuffers(1, (GLuint *)&mVBO);
}
