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
#include <memory>

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
class VRBServer
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
    

private:
    covise::ServerConnection *sConn = nullptr;
#ifdef GUI
    QSocketNotifier *serverSN = nullptr;
#endif
    covise::ConnectionList *connections = nullptr;
    int port; // port Number (default: 31800) covise.config: VRB.TCPPort
    void handleClient(covise::Message *);
    int createSession(bool isPrivate);
    std::shared_ptr<vrb::VrbServerRegistry> createSessionIfnotExists(int sessionID, int senderID);
    void sendSessions();
    void RerouteRequest(const char *location, int type, int senderId, int recvVRBId, QString filter, QString path);
    covise::Message *msg = nullptr;
    bool requestToQuit = false;
    VRBSClient *currentFileClient = nullptr;
    char *currentFile = nullptr;
    std::map<const int, std::shared_ptr<vrb::VrbServerRegistry>> sessions;
    std::string home();
};
#endif


