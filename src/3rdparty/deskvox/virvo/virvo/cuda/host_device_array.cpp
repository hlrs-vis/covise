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


#include "host_device_array.h"


#ifdef HAVE_CUDA


#include <cuda_runtime_api.h>

#include <string.h>

using virvo::cuda::HostDeviceArray;


void HostDeviceArray::reset()
{
    if (HostPtr)
        free(HostPtr);

    if (DevicePtr)
        cudaFree(DevicePtr);

    HostPtr = 0;
    DevicePtr = 0;
    Size = 0;
}


bool HostDeviceArray::resize(size_t count)
{
    reset();

    HostPtr = malloc(count);
    if (HostPtr == 0)
        return false;

    if (cudaSuccess != cudaMalloc(&DevicePtr, count))
    {
        free(HostPtr);
        HostPtr = 0;
        return false;
    }

    Size = count;

    return true;
}


bool HostDeviceArray::download()
{
    if (Size == 0)
        return true;

    return cudaSuccess == cudaMemcpy(HostPtr, DevicePtr, Size, cudaMemcpyDeviceToHost);
}


bool HostDeviceArray::upload()
{
    if (Size == 0)
        return true;

    return cudaSuccess == cudaMemcpy(DevicePtr, HostPtr, Size, cudaMemcpyHostToDevice);
}


void HostDeviceArray::fill(int value)
{
    if (Size == 0)
        return;

    // Clear the host buffer
    memset(HostPtr, value, Size);
    // Clear the device buffer
    cudaMemset(DevicePtr, value, Size);
}


#endif // HAVE_CUDA
