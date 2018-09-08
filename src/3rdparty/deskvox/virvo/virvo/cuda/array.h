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

#ifndef VV_CUDA_ARRAY_H
#define VV_CUDA_ARRAY_H

#include <cuda_runtime_api.h>


#if CUDART_VERSION < 5000

// CUDA array
typedef struct cudaArray *cudaArray_t;

// CUDA array (as source copy argument)
typedef const struct cudaArray *cudaArray_const_t;

#endif


namespace virvo
{
namespace cuda
{


    class Array
    {
        cudaArray_t arrayPtr;

    public:
        // Construct an empty array
        Array() : arrayPtr(0)
        {
        }

        // Clean up
        ~Array()
        {
            reset();
        }

        // Returns the pointer to the array.
        cudaArray_t get() { return arrayPtr; }

        // Returns the pointer to the array.
        cudaArray_const_t get() const { return arrayPtr; }

        // Sets the pointer to the array.
        // NOTE: Ownership of the array is transferred to this Array instance.
        void reset(cudaArray_t ptr = 0)
        {
            if (arrayPtr)
                cudaFreeArray(arrayPtr);

            arrayPtr = ptr;
        }

        // Release ownership of the currently held array
        cudaArray_t release()
        {
            cudaArray_t p = arrayPtr;
            arrayPtr = 0;
            return p;
        }

        // Allocate an array on the device.
        bool allocate(const cudaChannelFormatDesc& desc, size_t width, size_t height = 0, unsigned flags = 0)
        {
            reset();

            if (cudaSuccess == cudaMallocArray(&arrayPtr, &desc, width, height, flags))
                return true;

            arrayPtr = 0;

            return false;
        }

        // Allocate an array on the device.
        bool allocate3D(const cudaChannelFormatDesc& desc, const cudaExtent& extent, unsigned flags = 0)
        {
            reset();

            if (cudaSuccess == cudaMalloc3DArray(&arrayPtr, &desc, extent, flags))
                return true;

            arrayPtr = 0;

            return false;
        }

        // Allocate an array on the device.
        bool allocate3D(const cudaChannelFormatDesc& desc, size_t width, size_t height, size_t depth, unsigned flags = 0)
        {
            cudaExtent extent = { width, height, depth };

            return allocate3D(desc, extent, flags);
        }

        // Copy data from the device to the host.
        bool download(void* dst, size_t dst_pitch, size_t src_x, size_t src_y, size_t width, size_t height) const
        {
            return cudaSuccess == cudaMemcpy2DFromArray(dst, dst_pitch, get(), src_x, src_y, width, height, cudaMemcpyDeviceToHost);
        }

        // Copy data from the device to the host.
        bool download(void* dst, size_t src_x, size_t src_y, size_t count) const
        {
            return cudaSuccess == cudaMemcpyFromArray(dst, get(), src_x, src_y, count, cudaMemcpyDeviceToHost);
        }

        // Copy data from the host to the device.
        bool upload(size_t dst_x, size_t dst_y, const void* src, size_t src_pitch, size_t width, size_t height)
        {
            return cudaSuccess == cudaMemcpy2DToArray(get(), dst_x, dst_y, src, src_pitch, width, height, cudaMemcpyHostToDevice);
        }

        // Copy data from the host to the device.
        bool upload(size_t dst_x, size_t dst_y, const void* src, size_t count)
        {
            return cudaSuccess == cudaMemcpyToArray(get(), dst_x, dst_y, src, count, cudaMemcpyHostToDevice);
        }

    private:
        // NOT copyable!
        Array(Array const&);
        Array& operator =(Array const&);
    };


    // Copy data from device to device.
    inline bool copy(Array& dst, size_t dst_x, size_t dst_y, const Array& src, size_t src_x, size_t src_y, size_t width, size_t height)
    {
        return cudaSuccess == cudaMemcpy2DArrayToArray(dst.get(), dst_x, dst_y, src.get(), src_x, src_y, width, height, cudaMemcpyDeviceToDevice);
    }

    // Copy data from device to device.
    inline bool copy(Array& dst, size_t dst_x, size_t dst_y, const Array& src, size_t src_x, size_t src_y, size_t count)
    {
        return cudaSuccess == cudaMemcpyArrayToArray(dst.get(), dst_x, dst_y, src.get(), src_x, src_y, count, cudaMemcpyDeviceToDevice);
    }

    // Copy data from device to device.
    inline bool copy(void* dst, const Array& src, size_t src_x, size_t src_y, size_t count)
    {
        return cudaSuccess == cudaMemcpyFromArray(dst, src.get(), src_x, src_y, count, cudaMemcpyDeviceToDevice);
    }

    // Copy data from device to device.
    inline bool copy(Array& dst, size_t dst_x, size_t dst_y, const void* src, size_t count)
    {
        return cudaSuccess == cudaMemcpyToArray(dst.get(), dst_x, dst_y, src, count, cudaMemcpyDeviceToDevice);
    }


} // namespace cuda
} // namespace virvo

#endif // VV_CUDA_ARRAY_H
