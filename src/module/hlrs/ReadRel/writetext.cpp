/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "reldata.hpp"
#include "Globals.hpp"
#include <stdarg.h>
#include <stdio.h>

int WriteText(VERBOSE verb, const char *text, ...)
{
    va_list arglist;
    char buffer[1024];

    // Ellipse entschlüsseln
    va_start(arglist, text);
    vsprintf(buffer, text, arglist);
    va_end(arglist);

    if (Globals.verbose == VER_NONE)
        return (0);

    if (verb <= Globals.verbose)
    {
        printf("%s", buffer);
    }
    return (0);
}
