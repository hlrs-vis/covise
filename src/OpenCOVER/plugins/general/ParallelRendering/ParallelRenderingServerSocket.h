/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARALLELRENDERING_SERVER_SOCKET_H
#define PARALLELRENDERING_SERVER_SOCKET_H

#include "ParallelRenderingServer.h"

class ParallelRenderingServerSocket : public ParallelRenderingServer
{

public:
    ParallelRenderingServerSocket(int numClients, bool compositorRenders);
    virtual ~ParallelRenderingServerSocket();

    virtual void run();
    virtual void acceptConnection();
    virtual void render();

protected:
    int server_connect(int port);
    void receive();
    int *fd;

    OpenThreads::Mutex lock;
    OpenThreads::Mutex renderLock;
};

#endif
