/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef NDEBUG
#define CUDA_SAFE_CALL(FUNC) { cuda::safe_call((FUNC), __FILE__, __LINE__); }
#else
#define CUDA_SAFE_CALL(FUNC) FUNC
#endif
#define CUDA_SAFE_CALL_X(FUNC) { cuda::safe_call((FUNC), __FILE__, __LINE__, true); }

namespace cuda
{

inline void safe_call(cudaError_t code, char const* file, int line, bool fatal = false)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s %s:%i\n", cudaGetErrorString(code), file, line);
        if (fatal)
        {
            exit(code);
        }
    }
}

} // cuda


