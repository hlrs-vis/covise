/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SortLastImplementation.h"

#include <config/coConfigString.h>

SortLastImplementation::SortLastImplementation()
{
    covise::coConfigString commMethodEntry("method", "COVER.Parallel.SortLast.Comm");

    if (commMethodEntry == "gather")
        this->commMethod = Gather;
    else
        this->commMethod = Send;
}
