/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 *
 * Simple wrappers to make v2e running
 * R.M. 10.04.2001
 *
 * VirCinity(2001)
 */

#include <stdlib.h>

void *
RU_allocMem(size_t size, char *txt)
{
    return (malloc(size));
}

void
RU_freeMem(void *ptr)
{
    free(ptr);
}

void *
RU_reallocMem(void *ptr, size_t size)
{
    return (realloc(ptr, size));
}
