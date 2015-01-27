/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__UTIL_H)
#define __UTIL_H

#include <util/coviseCompat.h>

class WristWatch
{
private:
    timeval myClock;

public:
    WristWatch();
    ~WristWatch();

    void start();
    void stop(const char *s);
};
#endif // __UTIL_H
