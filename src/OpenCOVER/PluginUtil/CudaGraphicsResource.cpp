/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifdef HAVE_CUDA

#ifdef _WIN32
#include <windows.h> // APIENTRY
#endif
#include <cuda_gl_interop.h>

#include <CudaGraphicsResource.h>


namespace opencover
{

CudaGraphicsResource::CudaGraphicsResource()
    : m_resource(0)
    , m_devPtr(0)
{

}

CudaGraphicsResource::~CudaGraphicsResource()
{
    unregister();
}

cudaGraphicsResource_t CudaGraphicsResource::get() const
{
    return m_resource;
}

cudaError_t CudaGraphicsResource::register_buffer(unsigned buffer, cudaGraphicsRegisterFlags flags)
{
    unregister();
    return cudaGraphicsGLRegisterBuffer(&m_resource, buffer, flags);
}

cudaError_t CudaGraphicsResource::register_image(unsigned image, unsigned target, cudaGraphicsRegisterFlags flags)
{
    unregister();
    return cudaGraphicsGLRegisterImage(&m_resource, image, target, flags);
}

cudaError_t CudaGraphicsResource::unregister()
{
    if (m_resource == 0)
    {
        return cudaSuccess;
    }

    auto result = cudaGraphicsUnregisterResource(m_resource);
    m_resource = 0;
    return result;
}

void* CudaGraphicsResource::map(size_t* size)
{
    auto err = cudaGraphicsMapResources(1, &m_resource);
    if (err != cudaSuccess)
    {
        return 0;
    }

    err = cudaGraphicsResourceGetMappedPointer(&m_devPtr, size, m_resource);
    if (err != cudaSuccess)
    {
        cudaGraphicsUnmapResources(1, &m_resource);
        m_devPtr = 0;
    }

    return m_devPtr;
}

void* CudaGraphicsResource::map()
{
    size_t size = 0;
    return map(&size);
}

void CudaGraphicsResource::unmap()
{
    if (m_devPtr == 0)
    {
        return;
    }

    cudaGraphicsUnmapResources(1, &m_resource);
    m_devPtr = 0;
}

void* CudaGraphicsResource::dev_ptr() const
{
    return m_devPtr;
}

}

#endif
