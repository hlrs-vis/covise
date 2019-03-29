/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_MESAGE_HANDLER_H
#define VRB_MESAGE_HANDLER_H

#include <net/message.h>

#include <set>
#include <map>
#include <memory>
#include <string.h>
#include <vector>
#include <QString>
#ifdef GUI
#include <QSocketNotifier>
class ApplicationWindow;
#endif
class VRBSClient;
namespace covise
{
class Connection;
}

namespace vrb
{
class SessionID;
class VrbServerRegistry;

class ServerInterface
{
public:
    virtual void removeConnection(covise::Connection *conn) = 0;
#ifdef GUI
    virtual QSocketNotifier *getSN() = 0;
    virtual ApplicationWindow *getAW() = 0;
#endif // GUI

};
class VrbMessageHandler
{
public:
    VrbMessageHandler(ServerInterface *s);
    ///do stuff depening on message type
    void handleMessage(covise::Message *msg);
    ///inform clients about closing the socket
    void closeConnection();
    ///return numer of clients
    int numberOfClients();
#ifdef GUI
    ///create and add a new client
    void addClient(covise::Connection *conn, QSocketNotifier *sn);
    ///get the client corresponding to con and change its QSocketNotifier state; Return true if client exists
    bool setClientNotifier(covise::Connection *conn, bool state);
#endif 


private:
    ServerInterface *m_server;

    VRBSClient *currentFileClient = nullptr;
    char *currentFile = nullptr;

    std::map<vrb::SessionID, std::shared_ptr<vrb::VrbServerRegistry>> sessions;



    ///creates a session with the id in sessions, if id has no name, creates a name for it, if the id already exists returns
    vrb::SessionID & createSession(vrb::SessionID &id);
    ///Checs if the session already exists in sessionsion and if not, creates it and informs the sender
    std::shared_ptr<vrb::VrbServerRegistry>createSessionIfnotExists(vrb::SessionID &sessionID, int senderID);
    ///send a list of all public sessions to all clients
    void sendSessions();
    void RerouteRequest(const char *location, int type, int senderId, int recvVRBId, QString filter, QString path);

    ///return a path to the curren directory
    std::string home();
    ///get a kist of all files of type .fileEnding in the directory
    std::set<std::string> getFilesInDir(const std::string &path, const std::string &fileEnding = "")const;
    void disconectClientFromSessions(int clID);
    ///assign a client to a session
    void setSession(vrb::SessionID & sessionId);
};
}
#endif // !VRB_MESAGE_HANDLER_H
