// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
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

#ifndef VV_CUDA_MEMORY_H
#define VV_CUDA_MEMORY_H

#include <cuda_runtime_api.h>

#include <new> // bad_alloc


namespace virvo
{
namespace cuda
{


    template<class T>
    class AutoPointer
    {
        T* pointer;

    public:
        AutoPointer() : pointer(0)
        {
        }

        template<class U>
        explicit AutoPointer(U* pointer) : pointer(static_cast<T*>(pointer))
        {
        }

        // Clean up
        ~AutoPointer()
        {
            reset();
        }

        // Returns the currently held device pointer
        T* get() { return pointer; }

        // Returns the currently held device pointer
        const T* get() const { return pointer; }

        // Reset with the given pointer
        template<class U>
        void reset(U* newPointer)
        {
            if (pointer)
                cudaFree(pointer);

            pointer = static_cast<T*>(newPointer);
        }

        // Release memory
        void reset()
        {
            reset(static_cast<T*>(0));
        }

        // Release ownership of the currently held pointer
        T* release()
        {
            T* p = pointer;
            pointer = 0;
            return p;
        }

        // Allocate memory on the device
        bool allocate(size_t size)
        {
            reset();

            if (cudaSuccess == cudaMalloc(&pointer, size))
                return true;

            pointer = 0;

            return false;
        }

    private:
        // NOT copyable!
        AutoPointer(AutoPointer const&);
        AutoPointer& operator =(AutoPointer const&);
    };


    // Allocates size bytes of linear memory on the device.
    // Returns 0 on failure.
    inline void* deviceMalloc(size_t size)
    {
        void* devPtr = 0;

        if (cudaSuccess == cudaMalloc(&devPtr, size))
            return devPtr;

        return 0;
    }


    // Allocates size bytes of linear memory on the device.
    // Throws std::bad_alloc on failure.
    inline void* deviceNew(size_t size)
    {
        if (void* devPtr = deviceMalloc(size))
            return devPtr;

        throw std::bad_alloc();
    }


    // Copy memory from host to device
    template<class T, class U>
    inline bool upload(T* devTarget, const U* hostSource, size_t count)
    {
        return cudaSuccess == cudaMemcpy(devTarget, hostSource, count, cudaMemcpyHostToDevice);
    }


    // Copy memory from device to host
    template<class T, class U>
    inline bool download(T* hostTarget, const U* devSource, size_t count)
    {
        return cudaSuccess == cudaMemcpy(hostTarget, devSource, count, cudaMemcpyDeviceToHost);
    }


    // Copy memory from device to device
    template<class T, class U>
    inline bool copy(T* devTarget, const U* devSource, size_t count)
    {
        return cudaSuccess == cudaMemcpy(devTarget, devSource, count, cudaMemcpyDeviceToDevice);
    }


} // namespace cuda
} // namespace virvo

#endif // VV_CUDA_MEMORY_H

