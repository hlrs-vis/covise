/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Compatibility header for COVISE modules in YAC
#ifndef COVISE_COMPAT_H
#define COVISE_COMPAT_H

#include <errno.h>

#include "byteswap.h"
#include "coExport.h"
#include "common.h"

#if !(defined WIN32 || defined WIN64)
#include <sys/param.h>
# ifdef HZ
#  undef HZ
# endif
#endif
#include <cmath>
#include <iostream>
#include <sstream>

#endif
