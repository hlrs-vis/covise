/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARALLELRENDERING_CLIENT_Socket_H
#define PARALLELRENDERING_CLIENT_Socket_H

#include <virvo/vvsocketio.h>

#include "ParallelRenderingClient.h"

class ParallelRenderingClientSocket : public ParallelRenderingClient
{

public:
    ParallelRenderingClientSocket(int number, const std::string &compositor);
    virtual ~ParallelRenderingClientSocket();

    virtual void connectToServer();
    virtual void run();
    virtual void send();

private:
    int client_connect(const char *servername, int port);

    int fd;
    OpenThreads::Mutex lock;
    OpenThreads::Mutex sendLock;
};

#endif
