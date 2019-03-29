/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_SERVER_H
#define VRB_SERVER_H

#include <string>
#include <QObject>
#include <QString>
#include <map>
#include <set>
#include <memory>
#include <vrbclient/SessionID.h>
#include "VrbMessageHandler.h"
namespace covise
{
class ServerConnection;
class Connection;
class ConnectionList;
class Message;
}
class QSocketNotifier;
class VRBSClient;
namespace vrb
{
class VrbServerRegistry;
}

#ifdef Q_MOC_RUN
#define GUI
#endif
//
//
//
#ifndef GUI
class VRBServer : public vrb::ServerInterface
#else
class VRBServer : public QObject, public vrb::ServerInterface
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
    void removeConnection(covise::Connection *conn) override;
#ifdef  GUI
    QSocketNotifier *getSN() override;
    ApplicationWindow *getAW() override;
#endif //  GUI


private:
    covise::ServerConnection *sConn = nullptr;
    vrb::VrbMessageHandler *handler;
#ifdef GUI
    QSocketNotifier *serverSN = nullptr;
#endif
    covise::ConnectionList *connections = nullptr;
    int port; // port Number (default: 31800) covise.config: VRB.TCPPort
  
    covise::Message *msg = nullptr;
    bool requestToQuit = false;

};
#endif


