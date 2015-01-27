/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_WRIST_WATCH_H
#define COVISE_WRIST_WATCH_H

#include "coExport.h"
#include "unixcompat.h"

namespace covise
{

class UTILEXPORT coWristWatch
{
private:
    timeval myClock;

public:
    coWristWatch();

    void reset();
    float elapsed();
};
}
#endif
