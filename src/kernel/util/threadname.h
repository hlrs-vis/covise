/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef THREADNAME_H
#define THREADNAME_H

#include "coExport.h"
#include <string>

namespace covise
{
bool UTILEXPORT setThreadName(std::string name);
}

#endif // THREADNAME_H

