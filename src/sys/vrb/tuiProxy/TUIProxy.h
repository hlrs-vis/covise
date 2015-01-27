/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef YAC
#include "covise/covise.h"
#include <covise/covise_version.h>
#include <net/covise_connect.h>
#endif

namespace covise
{
class Connection;
class ConnectionList;
class ServerConnection;
class ClientConnection;
class Message;
}

class TUIProxy
{
public:
    TUIProxy(int argc, char **argv);
    ~TUIProxy();
    void handleMessages();
    int openServer();
    void closeServer();
    covise::ConnectionList *connections;

private:
    covise::ServerConnection *sConn;
    covise::ServerConnection *toCOVER;
    covise::ClientConnection *toTUI;
    int port;
};
