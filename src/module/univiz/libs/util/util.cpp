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
