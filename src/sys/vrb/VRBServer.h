/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_SERVER_H
#define VRB_SERVER_H

#include "coRegistry.h"
#include <util/coTabletUIMessages.h>

#include <QObject>
#include <QString>
#include <QStringList>

namespace covise
{
class ServerConnection;
class Connection;
class ConnectionList;
class Message;
}
class QSocketNotifier;
class VRBSClient;

//
//
//
#ifndef GUI
class VRBServer
#ifdef MOCONLY
{
};
#endif
#else
class VRBServer : public QObject
#endif
{

#ifdef GUI
    Q_OBJECT

private slots:
#endif
    void processMessages();

public:
    VRBServer();
    ~VRBServer();
    void loop();
    int openServer();
    void closeServer();
    coRegistry registry;

private:
    covise::ServerConnection *sConn;
    QSocketNotifier *serverSN;
    covise::ConnectionList *connections;
    int port; // port Number (default: 31800) covise.config: VRB.TCPPort
    void handleClient(covise::Message *);
    void RerouteRequest(const char *location, int type, int senderId, int recvVRBId, QString filter, QString path);
    covise::Message *msg;
    bool requestToQuit;
    VRBSClient *currentFileClient;
    char *currentFile;
};
#endif
