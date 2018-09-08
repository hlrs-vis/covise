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


#ifndef VV_CUDA_GRAPHICS_RESOURCE_H
#define VV_CUDA_GRAPHICS_RESOURCE_H


#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif


#ifdef HAVE_CUDA


#include "vvexport.h"

#include <cuda_runtime_api.h>


namespace virvo
{
namespace cuda
{


    class GraphicsResource
    {
    public:
        // Construct an empty graphics resource
        GraphicsResource()
            : Resource(0)
            , DevPtr(0)
        {
        }

        // Cleanup
        ~GraphicsResource()
        {
            unregister();
        }

        // Returns the CUDA resource
        cudaGraphicsResource_t get() const { return Resource; }

        // Register an OpenGL buffer object
        VVAPI bool registerBuffer(unsigned buffer, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);

        // Register an OpenGL image object
        VVAPI bool registerImage(unsigned image, unsigned target, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);

        // Unregister the currently held resource
        VVAPI void unregister();

        // Map the currently held resource into CUDA address space
        VVAPI void* map(size_t& size);

        // Map the currently held resource into CUDA address space
        VVAPI void* map();

        // Unmap the currently mapped resource
        VVAPI void unmap();

        // Returns a pointer to the mapped buffer data
        void* devPtr() const { return DevPtr; }

    private:
        // NOT copyable!
        GraphicsResource(GraphicsResource const&);
        GraphicsResource& operator =(GraphicsResource const&);

    private:
        // The graphics resource
        cudaGraphicsResource_t Resource;
        // Pointer into CUDA address space. Non-null if mapped, null otherwise
        void* DevPtr;
    };


} // namespace cuda
} // namespace virvo


#endif // HAVE_CUDA


#endif
