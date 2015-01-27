/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARALLELRENDERING_CLIENT_IBVERBS_H
#define PARALLELRENDERING_CLIENT_IBVERBS_H

#include "ParallelRenderingClient.h"
#include "IBVerbsTransport.h"

class ParallelRenderingClientIBVerbs : public ParallelRenderingClient
{

public:
    ParallelRenderingClientIBVerbs(int number, const std::string &compositor);
    virtual ~ParallelRenderingClientIBVerbs();

    virtual void run();
    virtual void send();

private:
    void connectToServer();

#ifdef HAVE_IBVERBS
    IBVerbsTransport *ib;
    Context *ctx;
    Destination *dest;
    Destination *remoteDest;
#endif
    int number;
    int once;
    int front;

    OpenThreads::Mutex lock;
};

#endif
