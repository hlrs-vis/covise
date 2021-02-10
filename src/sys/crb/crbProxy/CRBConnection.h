/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CRBCONNECTION_H
#define CRBCONNECTION_H

#include <covise/covise.h>
#include <net/covise_socket.h>
#ifndef NO_SIGNALS
#include <covise/covise_signal.h>
#endif
#include <covise/covise_msg.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <covise/covise_global.h>
#include "Proxy.h"

#ifndef _WIN32
#include <sys/param.h>
#endif


class CRBConnection
{
    char *host;
    int port;
    int id;
    std::list<Proxy *> modules;

public:
    covise::ConnectionList *listOfConnections; // list of all connections
    covise::Host *myHost;
    const covise::ServerConnection *toCrb;
    const covise::ClientConnection *toController;

    CRBConnection(int p, char *h, int id);
    ~CRBConnection();
    int execCRB(char *instance);
    void processMessages();

private:
    void contactController();
    void forwardMessage(covise::Message *msg, const covise::Connection *conn);
};
#endif
