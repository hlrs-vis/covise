/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARALLELRENDERING_SERVER_IBVERBS_H
#define PARALLELRENDERING_SERVER_IBVERBS_H

#include "ParallelRenderingServer.h"
#include "IBVerbsTransport.h"

class ParallelRenderingServerIBVerbs : public ParallelRenderingServer
{

public:
    ParallelRenderingServerIBVerbs(int numClients, bool compositorRenders);
    virtual ~ParallelRenderingServerIBVerbs();

    virtual void run();
    virtual void render();
    virtual void acceptConnection();

protected:
    void receive();

#ifdef HAVE_IBVERBS
    IBVerbsTransport *ib;
    Context **ctx;
    Destination **dest;
    Destination **remoteDest;
#endif
    int width;
    int height;

    OpenThreads::Mutex lock;
    OpenThreads::Mutex renderLock;
    int front;
};

#endif
