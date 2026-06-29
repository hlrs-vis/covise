/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _STAR_H
#define _STAR_H

#include <stdlib.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <api/coModule.h>
using namespace covise;

// Example module for Covise API 2.0 User-interface functions
class TestSoc : public coModule
{

private:
    //////////  member functions
    virtual void sockData(int soc);
    virtual void postInst();

    // our socket
    int fd1, fd2, fd3;

public:
    TestSoc(int argc, char *argv[]);
};
#endif // _READSTAR_H
