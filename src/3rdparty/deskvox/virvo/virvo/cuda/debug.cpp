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


#include "debug.h"


#ifdef HAVE_CUDA


#ifdef _MSC_VER
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>

namespace cu = virvo::cuda;


void cu::debug_report_error(cudaError_t err, char const* file, int line)
{
    fprintf(stderr, "%s(%d) : CUDA error: %s\n", file, line, cudaGetErrorString(err));

#ifdef _MSC_VER
    if (IsDebuggerPresent())
    {
        DebugBreak();
    }
#else
    //exit(EXIT_FAILURE);
#endif
}


void cu::debug_meminfo(char const* file, int line)
{
    size_t avail = 0;
    size_t total = 0;

    cudaError_t err = cudaMemGetInfo(&avail, &total);
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%d) : CUDA memory usage: cudaMemGetInfo failed: %s\n", file, line, cudaGetErrorString(err));
        return;
    }

    double davail = static_cast<double>(avail);
    double dtotal = static_cast<double>(total);

    fprintf(stdout, "%s(%d) : CUDA memory usage: %e%%\n", file, line, 100.0 * (davail / dtotal));
}

#endif // HAVE_CUDA
