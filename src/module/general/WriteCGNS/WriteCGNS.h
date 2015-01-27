/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_WRITECGNS_H
#define CO_WRITECGNS_H 1

#include <api/coSimpleModule.h>
#include <util/coviseCompat.h>
#include <cgnslib.h>
#include <vector>

using namespace covise;

class WriteCGNS : public coSimpleModule
{
    int cgnsFile;

    coInputPort *gridPort;
    virtual int compute(const char *port);

    coFileBrowserParam *m_FileNameParam;

public:
    WriteCGNS(int argc, char *argv[]);
    virtual ~WriteCGNS();
};
#endif
