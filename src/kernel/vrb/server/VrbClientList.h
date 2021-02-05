/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRBCLIENTLIST_H
#define VRBCLIENTLIST_H

#include <net/covise_connect.h>
#include <net/message_types.h>
#include <util/coExport.h>
#include <vrb/RemoteClient.h>
#include <vrb/SessionID.h>
#include <vrb/UserInfo.h>
#include <net/message_sender_interface.h>

#include <string>
#include <set>
#include <vector>
namespace covise
{
class TokenBuffer;
class Message;
class UdpMessage;
class MessageBase;
}

namespace vrb
{
class ServerInterface;

enum VRBSERVEREXPORT Columns {
    Master,
    ID,
    Group,
    User,
    Host,
    Email,
    URL,
    IP,
};

class VRBSERVEREXPORT VRBSClient : public vrb::RemoteClient, public covise::MessageSenderInterface
{
	///Vrb Server client that holds a connection and information about the client
public:

    ///send = false if you dont want to inform the client immediatly about the contact
    VRBSClient(covise::Connection *c, covise::UDPConnection* udpc, covise::TokenBuffer &tb, bool deleteClient = true);
    virtual ~VRBSClient();

    covise::Connection *conn = nullptr;
	covise::UDPConnection* udpConn = nullptr;

    const vrb::SessionID &getPrivateSession() const;
    void setPrivateSession(vrb::SessionID &g);
    int getSentBPS();
    int getReceivedBPS();
    void setInterval(float i);
    void addBytesSent(int b);
    void addBytesReceived(int b);
	bool doesNotKnowFile(const std::string& fileName);
	void addUnknownFile(const std::string& fileName);
private:
    bool sendMessage(const covise::Message *msg) const override;
    bool sendMessage(const covise::UdpMessage *msg) const override;

protected:
    std::set<std::string> m_unknownFiles;
    vrb::SessionID m_privateSession;
    long m_bytesSent = 0;
    long m_bytesReceived = 0;
    double m_lastRecTime = 0.0;
    double m_lastSendTime = 0.0;
    float m_interval = 0;
    int m_bytesSentPerInterval = 0;
    int m_bytesReceivedPerInterval = 0;
    int m_bytesSentPerSecond = 0;
    int m_bytesReceivedPerSecond = 0;
	bool m_deleteClient = true;
	mutable bool m_firstTryUdp = true;
    double time();

};

class VRBSERVEREXPORT VRBClientList
{
protected:
    std::vector<std::unique_ptr<VRBSClient>> m_clients;

public:
    VRBClientList() = default;
    virtual ~VRBClientList() = default;

    VRBClientList(const VRBClientList&) = delete;
    VRBClientList(VRBClientList&&) = delete;
    VRBClientList& operator=(const VRBClientList&) = delete;
    VRBClientList& operator=(VRBClientList&&) = delete;

	VRBSClient *get(covise::Connection *c);
    VRBSClient *get(const char *ip);
    VRBSClient *get(int id);
    VRBSClient *getMaster(const vrb::SessionID &session);
    VRBSClient *getNextInGroup(const vrb::SessionID &id);
    VRBSClient *getNthClient(size_t N);
    std::vector<VRBSClient *> getClientsWithUserName(const std::string &name);
    int getNextFreeClientID();
    ///client becomes master and all other clients in clients in session lose master state
    void setMaster(VRBSClient *client);
    void setInterval(float i);
    void deleteAll();
    ///send mesage to every member of the session
    void sendMessage(covise::TokenBuffer &stb, const vrb::SessionID &group = vrb::SessionID(0, "all", false), covise::covise_msg_type type = covise::COVISE_MESSAGE_VRB_GUI);
    ///send message to the client with id
    void sendMessageToClient(VRBSClient *cl, covise::TokenBuffer &tb, covise::covise_msg_type type);
    void sendMessageToClient(int clientID, covise::TokenBuffer &stb, covise::covise_msg_type type = covise::COVISE_MESSAGE_VRB_GUI);
    void sendMessageToAll(covise::TokenBuffer &tb, covise::covise_msg_type type = covise::COVISE_MESSAGE_VRB_GUI);
	static std::string cutFileName(const std::string& fileName);
    int numInSession(vrb::SessionID &Group);
    size_t numberOfClients();
    void addClient(VRBSClient *cl);
    void removeClient(VRBSClient *cl);
	/// remove client with connection c
	void remove(covise::Connection *c);
    /// pass Message to all other session participants 
    void passOnMessage(const covise::MessageBase* msg, const vrb::SessionID &session = vrb::SessionID(0, "", false));
    /// bradcast the message to all clients of the progam type
    void broadcastMessageToProgramm(vrb::Program program, covise::MessageBase *msg);
    ///write the info of all clients in the tokenbuffer
    void collectClientInfo(covise::TokenBuffer &tb, const VRBSClient *recipient) const;
    VRBSClient *getNextPossibleFileOwner(const std::string &fileName, const vrb::SessionID &id);
};
extern VRBSERVEREXPORT VRBClientList clients;

}
#endif