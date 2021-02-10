/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ACCESSGRID_DAEMON_H
#define ACCESSGRID_DAEMON_H

namespace covise
{
class ServerConnection;
class SimpleServerConnection;
class Connection;
class ConnectionList;
class CoviseConfig;
class Message;
}

class QSocketNotifier;

//
//
//
class AccessGridDaemon
{

public:
    AccessGridDaemon();
    ~AccessGridDaemon();
    void loop();
    int openServer();
    void closeServer();
    void startCovise();

private:
    const covise::ServerConnection *sConn;
    const covise::ServerConnection *toController;
    const covise::Connection *toAG;
    covise::ConnectionList *connections;
    int port; // port Number (default: 31098) covise.config: ACCESSGRID_DAEMON.TCPPort
    covise::Message *msg;

    void handleClient(covise::Message *);
    int handleClient(const char *, const covise::Connection *conn);
    int processMessages();
};

#endif
