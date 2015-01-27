/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>

void my_fatal(const char *src, int line, const char *text);

#define fatal(x) my_fatal(__FILE__, __LINE__, (x))
