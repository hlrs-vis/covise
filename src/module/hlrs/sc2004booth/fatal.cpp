/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "include/fatal.h"
#include "include/log.h"

void my_fatal(const char *src, int line, const char *text)
{
    dprintf(0, (char *)"source: %s\n", src);
    dprintf(0, (char *)"line  : %d\n", line);
    dprintf(0, (char *)"bug   : %s\n", text);
    exit(1);
}
