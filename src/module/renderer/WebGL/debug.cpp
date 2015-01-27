/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "debug.h"
#include <stdio.h>

extern int msglevel; /* the higher, the more messages... */

#if defined(NDEBUG) && defined(__GNUC__)
/* Nothing. pmesg has been "defined away" in debug.h already. */
#else
void pmesg(int level, const char *format, ...)
{
#ifdef NDEBUG
/* Empty body, so a good compiler will optimise calls
           to pmesg away */
#else
    va_list args;

    if (level > msglevel)
        return;

    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
#endif /* NDEBUG */
}
#endif /* NDEBUG && __GNUC__ */
