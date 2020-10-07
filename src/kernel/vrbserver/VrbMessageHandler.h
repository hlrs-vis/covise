/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///base class to process vrb messages used in the vrb application and COVISE for the communication between OpenCOVERs

#ifndef VRB_MESAGE_HANDLER_H
#define VRB_MESAGE_HANDLER_H
#include "VrbSessionList.h"
#include <net/message.h>
#include <util/coExport.h>
#include <vrbserver/VrbClientList.h>
#include <set>
#include <map>
#include <unordered_map>
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
class DataHandle;
}

namespace vrb
{
class VRBSClient;
class SessionID;
class VrbServerRegistry;
class UdpMessage;

class VRBSERVEREXPORT ServerInterface
{
public:
    virtual void removeConnection(covise::Connection *conn) = 0;

};

class VRBSERVEREXPORT VrbMessageHandler
{
public:
    VrbMessageHandler(ServerInterface *server);
    void handleMessage(covise::Message *msg);
	void handleUdpMessage(UdpMessage* msg);
    int numberOfClients();
    void addClient(VRBSClient* client);
    void remove(covise::Connection* c);
    void closeConnection();
protected:
	///update the vrb userinterface
    virtual void updateApplicationWindow(const std::string& cl, int sender, const std::string& var, const covise::DataHandle &value);
    virtual void removeEntryFromApplicationWindow(const std::string& cl, int sender, const std::string& var);
    virtual void removeEntriesFromApplicationWindow(int sender);

private:
    ServerInterface *m_server = nullptr;
    VrbSessionList sessions;
	///stack of sessions that have to be set at the client after he received the userdata of all clients
	std::set<int> m_sessionsToSet;
    //participants: clients in a session
    //return: send back to sender
    void handleFileRequest(covise::Message* msg, covise::TokenBuffer& tb);
    void sendFile(covise::Message* msg, covise::TokenBuffer& tb);
    void changeRegVar(covise::TokenBuffer& tb);
    VrbServerRegistry& createSessionIfnotExists(const SessionID &sessionID, int senderID);
    void handleNewClient(covise::TokenBuffer& tb, covise::Message* msg);
    void setUserInfo(covise::TokenBuffer& tb, covise::Message* msg);
    void informClientsAboutNewClient(vrb::VRBSClient* c);
    void informNewClientAboutClients(vrb::VRBSClient* c);
    void setNewClientsStartingSession(vrb::VRBSClient* c);
    void passMessageToParticipants(covise::Message* msg);
    void returnWhetherClientExists(covise::TokenBuffer& tb, covise::Message* msg);
    void returnStartSession(covise::Message* msg);
    void determineNewMaster(vrb::SessionID& newSession, vrb::VRBSClient* c);
    void informParticipantsAboutNewMaster(vrb::VRBSClient* c);
    void informParticipantsAboutNewParticipant(vrb::VRBSClient* c);
    void sendSessions();
    void setSession(const SessionID & sessionId, int clID);
    void returnSerializedSessionInfo(covise::TokenBuffer& tb);
    void loadSession(const covise::DataHandle& file, const SessionID &currentSession);
    void informParticipantsAboutLoadedSession(int senderID, SessionID sid);
};
}
#endif // !VRB_MESAGE_HANDLER_H
