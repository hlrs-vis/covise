/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TILED_DISPLAY_CLIENT_IBVERBS_H
#define TILED_DISPLAY_CLIENT_IBVERBS_H

#include "TiledDisplayClient.h"
#include "IBVerbsTransport.h"

class TiledDisplayClientIBVerbs : public TiledDisplayClient
{

public:
    TiledDisplayClientIBVerbs(int number, const std::string &compositor);
    virtual ~TiledDisplayClientIBVerbs();

    virtual bool connect();
    virtual void run();

private:
#ifdef HAVE_IBVERBS
    IBVerbsTransport *ib;
    Context *ctx;
    Destination *dest;
    Destination *remoteDest;
#endif
    int number;
    int once;
};

#endif
