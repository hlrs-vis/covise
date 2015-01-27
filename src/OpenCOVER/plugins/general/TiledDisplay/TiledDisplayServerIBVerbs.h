/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_SERVER_IBVERBS_H
#define TILED_DISPLAY_SERVER_IBVERBS_H

#include "TiledDisplayServer.h"
#include "IBVerbsTransport.h"

class TiledDisplayServerIBVerbs : public TiledDisplayServer
{

public:
    TiledDisplayServerIBVerbs(int number);
    virtual ~TiledDisplayServerIBVerbs();

    bool accept();
    void run();

protected:
#ifdef HAVE_IBVERBS
    IBVerbsTransport *ib;
    Context *ctx;
    Destination *dest;
    Destination *remoteDest;
#endif
    int number;
    int width;
    int height;
    int once;
};

#endif
