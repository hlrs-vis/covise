/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifdef HAVE_CUDA

#ifndef CUDAGRAPHICSRESOURCE_H
#define CUDAGRAPHICSRESOURCE_H

#include <cuda_runtime_api.h>

#include <util/coExport.h>


namespace opencover
{

class CudaGraphicsResource
{
public:

    CudaGraphicsResource();
    ~CudaGraphicsResource();

    CudaGraphicsResource(CudaGraphicsResource&) = delete;
    CudaGraphicsResource& operator=(CudaGraphicsResource&) = delete;

    cudaGraphicsResource_t get() const;

    cudaError_t register_buffer(unsigned buffer, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    cudaError_t register_image(unsigned image, unsigned target, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    cudaError_t unregister();

    void* map(size_t* size);
    void* map();
    void unmap();

    void* dev_ptr() const;

private:
    cudaGraphicsResource_t m_resource;
    void* m_devPtr;

};

}

#endif

#endif
