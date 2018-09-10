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


#ifndef VV_CUDA_HOST_DEVICE_ARRAY_H
#define VV_CUDA_HOST_DEVICE_ARRAY_H


#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif


#ifdef HAVE_CUDA


#include "vvexport.h"

#include <stdlib.h>
#include <new> // bad_alloc


namespace virvo
{
namespace cuda
{


    class HostDeviceArray
    {
    public:
        // Construct an empty buffer
        HostDeviceArray()
            : Size(0)
            , HostPtr(0)
            , DevicePtr(0)
        {
        }

        // Construct a buffer with the given size
        HostDeviceArray(size_t count)
            : Size(0)
            , HostPtr(0)
            , DevicePtr(0)
        {
            if (!resize(count))
                throw std::bad_alloc();
        }

        // Clean up
        ~HostDeviceArray()
        {
            reset();
        }

        // Returns the size in bytes of the host and the device array
        size_t size() const { return Size; }

        // Returns a pointer to the host data
        void* hostPtr() const { return HostPtr; }

        // Returns a pointer to the device data
        void* devicePtr() const { return DevicePtr; }

        // Release all resources
        VVAPI void reset();

        // Resize the host and device arrays
        VVAPI bool resize(size_t count);

        // Download the contents of the device array into the host array
        VVAPI bool download();

        // Upload the contents of the host array into the device array
        VVAPI bool upload();

        // Performs a memset on the host and device buffers
        VVAPI void fill(int value);

    private:
        // NOT copyable!
        HostDeviceArray(HostDeviceArray const&);
        HostDeviceArray& operator =(HostDeviceArray const&);

    private:
        // The size of the host and the device buffer
        size_t Size;
        // Pointer to the host data
        void* HostPtr;
        // Pointer to the device data
        void* DevicePtr;
    };


} // namespace cuda
} // namespace virvo


#endif // HAVE_CUDA


#endif
