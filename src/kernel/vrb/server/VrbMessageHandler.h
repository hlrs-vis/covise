/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 ///base class to process vrb messages used in the vrb application and COVISE for the communication between OpenCOVERs

#ifndef VRB_MESAGE_HANDLER_H
#define VRB_MESAGE_HANDLER_H
#include "VrbSessionList.h"
#include <net/message.h>
#include <net/covise_connect.h>
#include <util/coExport.h>
#include "VrbClientList.h"
#include <set>
#include <map>
#include <unordered_map>
#include <memory>
#include <string.h>
#include <vector>

#ifdef Q_MOC_RUN
#define GUI
#endif

namespace covise
{
class DataHandle;
class UdpMessage;
}

namespace vrb
{
class VRBSClient;
class SessionID;
class VrbServerRegistry;

struct VRBSERVEREXPORT ConnectionDetails {
	typedef std::unique_ptr<ConnectionDetails> ptr;

	const covise::Connection *tcpConn = nullptr;
	const covise::UDPConnection* udpConn = nullptr;
	ConnectionDetails() = default;
	ConnectionDetails(ConnectionDetails& other) = delete;
	virtual ConnectionDetails& operator=(ConnectionDetails& other) = delete;
	ConnectionDetails(ConnectionDetails&& other) = default;
	virtual ConnectionDetails& operator=(ConnectionDetails&& other) = default;
	virtual ~ConnectionDetails() = default;
};
class VRBSERVEREXPORT ServerInterface
{
public:
	virtual void removeConnection(const covise::Connection* conn) = 0;

};

class VRBSERVEREXPORT VrbMessageHandler
{
public:
	VrbMessageHandler(ServerInterface* server);

	VrbMessageHandler(VrbMessageHandler& other) = delete;
	virtual VrbMessageHandler& operator=(VrbMessageHandler& other) = delete;
	VrbMessageHandler(VrbMessageHandler&& other) = delete;
	virtual VrbMessageHandler& operator=(VrbMessageHandler&& other) = delete;
	virtual ~VrbMessageHandler() = default;

	void handleMessage(covise::Message* msg);
	void handleUdpMessage(covise::UdpMessage* msg);
	int numberOfClients();
	void addClient(ConnectionDetails::ptr&& clientCon);
	void closeConnection();
	void remove(const covise::Connection* c);
protected:
	///update the vrb userinterface
	virtual void updateApplicationWindow(const std::string& cl, int sender, const std::string& var, const covise::DataHandle& value);
	virtual void removeEntryFromApplicationWindow(const std::string& cl, int sender, const std::string& var);
	virtual void removeEntriesFromApplicationWindow(int sender);
	virtual VRBSClient* createNewClient(ConnectionDetails::ptr&& cd, covise::TokenBuffer& tb);

private:
	ServerInterface* m_server = nullptr;
	VrbSessionList m_sessions;
	///stack of sessions that have to be set at the client after he received the userdata of all clients
	std::set<int> m_sessionsToSet;
	std::vector<ConnectionDetails::ptr> m_unregisteredClients;
	void removeUnregisteredClient(const covise::Connection* conn);
	VRBSClient* createNewClient(covise::TokenBuffer& tb, covise::Message* msg);
	//participants: clients in a session
	//return: send back to sender
	void handleFileRequest(covise::Message* msg, covise::TokenBuffer& tb);
	void sendFile(covise::Message* msg, covise::TokenBuffer& tb);
	void changeRegVar(covise::TokenBuffer& tb);
	VrbServerRegistry& putClientInSession(const SessionID& sessionID, int senderID);
	VrbServerRegistry& putClientInSession(const SessionID& sessionID, VRBSClient* cl);

	void handleNewClient(covise::TokenBuffer& tb, covise::Message* msg);
	void informClientsAboutNewClient(VRBSClient* c);
	void informNewClientAboutClients(VRBSClient* c);
	void setNewClientsStartingSession(VRBSClient* c);
	void passMessageToParticipants(covise::Message* msg);
	void returnWhetherClientExists(covise::TokenBuffer& tb, covise::Message* msg);
	void returnStartSession(covise::Message* msg);
	void determineNewMaster(const SessionID& sid);
	void informOtherParticipantsAboutNewMaster(VRBSClient* c);
	void informOtherParticipantsAboutNewParticipant(VRBSClient* c);
	void sendSessions();
	void setSession(const SessionID& sessionId, int clID);
	void returnSerializedSessionInfo(covise::TokenBuffer& tb);
	void loadSession(const covise::DataHandle& file, const SessionID& currentSession);
	void informParticipantsAboutLoadedSession(int senderID, SessionID sid);
};
}
#endif // !VRB_MESAGE_HANDLER_H
