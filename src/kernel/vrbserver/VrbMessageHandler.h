/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_MESAGE_HANDLER_H
#define VRB_MESAGE_HANDLER_H

#include <net/message.h>
#include <util/coExport.h>
#include <vrbserver/VrbClientList.h>
#include <set>
#include <map>
#include <memory>
#include <string.h>
#include <vector>
#include <QString>
#ifdef Q_MOC_RUN
#define GUI
#endif

namespace covise
{
class Connection;
}

namespace vrb
{
class VRBSClient;
class SessionID;
class VrbServerRegistry;

class VRBSERVEREXPORT ServerInterface
{
public:
    virtual void removeConnection(covise::Connection *conn) = 0;

};

class VRBSERVEREXPORT VrbMessageHandler
{
public:
    VrbMessageHandler(ServerInterface *server);
    ///do stuff depening on message type
    virtual void handleMessage(covise::Message *msg);
    ///
    virtual  
    ///inform clients about closing the socket
    void closeConnection();
    ///return numer of clients
    int numberOfClients();

    ///add a client
    void addClient(VRBSClient *client);

	/// remove client with connection c
	void remove(covise::Connection *c); 


protected:
    ServerInterface *m_server;
    virtual void updateApplicationWindow(const char *cl, int sender, const char *var, covise::TokenBuffer &value);
    virtual void removeEntryFromApplicationWindow(const char *cl, int sender, const char *var);
    virtual void removeEntriesFromApplicationWindow(int sender);

    char *currentFile = nullptr;

    const std::string suffix = ".vrbreg";

    std::map<vrb::SessionID, std::shared_ptr<vrb::VrbServerRegistry>> sessions;



    ///changes the given ID to a unique sessionID as close as possible to the given ID; Private/Publi state keeps unchanged
    ///Create a new entry in sessions map for the new sessionID
    void createSession(vrb::SessionID &id);
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
    void setSession(vrb::SessionID & sessionId, int clID);
    ///writes the session "id" and the private sessions of id's participants to disc
    void saveSession(const vrb::SessionID & id);
    ///load a session from disc into currentSession 
    void loadSesion(const std::string & name, const vrb::SessionID &currentSession);
    ///get current time as a formatted string for filenames
    std::string getTime() const;
    ///stores new loaded files in the fileOwner for later reference when a file needs to be fetched
    void storeLoadedFiles(covise::TokenBuffer & tb, VRBSClient *fileOwner);
    ///
};
}
#endif // !VRB_MESAGE_HANDLER_H
