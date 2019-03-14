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
    ///creates a session with the id in sessions, if id has no name, creates a name for it, if the id already exists returns
    vrb::SessionID & createSession(vrb::SessionID &id);
    ///Checs if the session already exists in sessionsion and if not, creates it and informs the sender
    std::shared_ptr<vrb::VrbServerRegistry>createSessionIfnotExists(vrb::SessionID &sessionID, int senderID);
    ///send a list of all public sessions to all clients
    void sendSessions();
    void RerouteRequest(const char *location, int type, int senderId, int recvVRBId, QString filter, QString path);
    covise::Message *msg = nullptr;
    bool requestToQuit = false;
    VRBSClient *currentFileClient = nullptr;
    char *currentFile = nullptr;
    std::map<vrb::SessionID, std::shared_ptr<vrb::VrbServerRegistry>> sessions;
    ///return a path to the curren directory
    std::string home();
    ///get a kist of all files of type .fileEnding in the directory
    std::set<std::string> getFilesInDir(const std::string &path, const std::string &fileEnding = "")const;
    void disconectClientFromSessions(VRBSClient * cl);
    ///assign a client to a session
    void setSession(vrb::SessionID & sessionId);
};
#endif


