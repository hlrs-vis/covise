/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "coExport.h"
#include <string>

namespace covise
{
bool UTILEXPORT setupEnvironment(int argc, char *argv[]);
#ifdef __APPLE__
std::string UTILEXPORT getBundlePath();
#endif
}

#endif // ENVIRONMENT_H
