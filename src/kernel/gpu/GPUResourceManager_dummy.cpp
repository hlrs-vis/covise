/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GPUResourceManager.h"

GPUUsg *GPUResourceManager::allocUSG(const char *, int, int, int,
                                     int *, int *, int *,
                                     float *, float *, float *)
{
    return NULL;
}

GPUScalar *GPUResourceManager::allocScalar(const char *, int, float *)
{
    return NULL;
}

void GPUResourceManager::deallocUSG(GPUUsg *)
{
}

void GPUResourceManager::deallocScalar(GPUScalar *)
{
}
