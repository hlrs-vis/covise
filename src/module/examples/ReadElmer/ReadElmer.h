/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_ELMER_H
#define _READ_ELMER_H

#include <stdlib.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <api/coModule.h>
using namespace covise;

class ReadElmer : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);

    coOutputPort *meshOutPort;
    coOutputPort *velOutPort;
    coOutputPort *pressOutPort;
    coOutputPort *keOutPort;

    coFileBrowserParam *filenameParam;

public:
    ReadElmer(int argc, char *argv[]);
    virtual ~ReadElmer();
};
#endif
