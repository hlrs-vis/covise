// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2012 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA


#include "graphics_resource.h"


#ifdef HAVE_CUDA


#include "debug.h"


#ifdef _WIN32
#include <windows.h> // APIENTRY
#endif
#include <assert.h>
#include <cuda_gl_interop.h>


using virvo::cuda::GraphicsResource;


bool GraphicsResource::registerBuffer(unsigned buffer, cudaGraphicsRegisterFlags flags)
{
    unregister();
    return cudaSuccess == cudaGraphicsGLRegisterBuffer(&Resource, buffer, flags);
}


bool GraphicsResource::registerImage(unsigned image, unsigned target, cudaGraphicsRegisterFlags flags)
{
    unregister();
    return cudaSuccess == cudaGraphicsGLRegisterImage(&Resource, image, target, flags);
}


void GraphicsResource::unregister()
{
    if (Resource == 0)
        return;

    cudaGraphicsUnregisterResource(Resource);
    Resource = 0;
}


void* GraphicsResource::map(size_t& size)
{
    if (cudaSuccess != cudaGraphicsMapResources(1, &Resource))
        return 0;

    if (cudaSuccess != cudaGraphicsResourceGetMappedPointer(&DevPtr, &size, Resource))
    {
        cudaGraphicsUnmapResources(1, &Resource);
        DevPtr = 0;
    }

    return DevPtr;
}


void* GraphicsResource::map()
{
    size_t size = 0;

    return map(size);
}


void GraphicsResource::unmap()
{
    if (DevPtr == 0)
        return;

    cudaGraphicsUnmapResources(1, &Resource);
    DevPtr = 0;
}


#endif // HAVE_CUDA
