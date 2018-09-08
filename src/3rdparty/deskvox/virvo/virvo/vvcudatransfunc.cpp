// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 2010 University of Cologne
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


#include "vvcudatransfunc.h"


#ifdef HAVE_CUDA


#include "cuda/memory.h"

namespace cu = virvo::cuda;


// vvcudatransfunc.cu
extern "C" void CallCudaTransFuncKernel(int width, uchar4* preIntTable, float thickness, float min, float max, const float4* rgba);


bool makePreintLUTCorrectCuda(int width, unsigned char *preIntTable, float thickness, float min, float max, const float *rgba)
{
    cu::AutoPointer<uchar4> d_preIntTable ( cu::deviceMalloc(sizeof(uchar4) * width * width) );
    cu::AutoPointer<float4> d_rgba        ( cu::deviceMalloc(sizeof(float4) * (width + 1)) );

    if (!d_preIntTable.get() || !d_rgba.get())
        return false;

    cu::upload(d_rgba.get(), rgba, sizeof(float4) * (width + 1));

    CallCudaTransFuncKernel(width, d_preIntTable.get(), thickness, min, max, d_rgba.get());

    cu::download(preIntTable, d_preIntTable.get(), sizeof(uchar4) * width * width);

    return true;
}


#endif // HAVE_CUDA
