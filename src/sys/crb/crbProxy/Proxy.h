/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PROXIE_H
#define PROXIE_H

#include <covise/covise.h>
#include <net/covise_socket.h>
#ifndef NO_SIGNALS
#include <covise/covise_signal.h>
#endif
#include <covise/covise_msg.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <covise/covise_global.h>

#ifndef _WIN32
#include <sys/param.h>
#endif
class CRBConnection;

class Proxy
{
public:
    CRBConnection *crbConn;
    covise::ServerConnection *moduleConn; // connection to module
    covise::ClientConnection *ctrlConn; // connection to controller
    Proxy(char *messageData, CRBConnection *);
    ~Proxy();

private:
};
#endif
