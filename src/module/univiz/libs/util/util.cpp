/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Unification Library for Modular Visualization Systems
//
// Utilities
//
// CGL ETH Zuerich
// Filip Sadlo 2008 -

#include "util.h"
#include <stdio.h>

bool fileReadable(const char *fileName)
{
    FILE *fp;
    bool readable = false;

    fp = fopen(fileName, "r");
    if (fp)
    {
        readable = true;
        fclose(fp);
    }
    return readable;
}
