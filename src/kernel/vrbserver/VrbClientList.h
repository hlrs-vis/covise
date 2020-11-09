/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRBCLIENTLIST_H
#define VRBCLIENTLIST_H
#include <util/coExport.h>
#include <vrbclient/SessionID.h>
#include <net/message_types.h>
#include <string>
#include <set>
#include <vector>
#include <vrbclient/UserInfo.h>
namespace covise
{
class TokenBuffer;
class Connection;
class UDPConnection;
class Message;
class MessageBase;
}

namespace vrb
{
class ServerInterface;
class UdpMessage;

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

class VRBSERVEREXPORT VRBSClient
{
	///Vrb Server client that holds a connection and information about the client
public:

    ///send = false if you dont want to inform the client immediatly about the contact
    VRBSClient(covise::Connection *c, covise::UDPConnection* udpc, const char *ip, const char *name , bool send = true, bool deleteClient = true);
    virtual ~VRBSClient();
    ///set clientinformation and inform the client about its server id and session
    virtual void setContactInfo(const char *ip, const char *n, vrb::SessionID &session);
    ///store userinfo like email, pc-name, ...
    virtual void setUserInfo(const UserInfo& userInfo);
    covise::Connection *conn = nullptr;
	covise::UDPConnection* udpConn = nullptr;
	void sendMsg(covise::MessageBase* msg);
    const std::string &getName() const;
    const std::string &getIP() const;
    int getID() const;
    const vrb::SessionID &getSession() const;
    virtual void setSession(const vrb::SessionID &g);
    const vrb::SessionID &getPrivateSession() const;
    void setPrivateSession(vrb::SessionID &g);
    bool isMaster();
    virtual void setMaster(bool m);
    UserInfo getUserInfo();
    std::string getUserName();
    int getSentBPS();
    int getReceivedBPS();
    void setInterval(float i);
    void addBytesSent(int b);
    void addBytesReceived(int b);
    void getInfo(covise::TokenBuffer &rtb);
	bool doesNotKnowFile(const std::string& fileName);
	void addUnknownFile(const std::string& fileName);


protected:
    std::set<std::string> m_unknownFiles;
    std::string m_name;
    UserInfo userInfo;
    int myID = -1;
    vrb::SessionID m_publicSession, m_privateSession;
    bool m_master = false;
    long bytesSent = 0;
    long bytesReceived = 0;
    double lastRecTime = -1.;
    double lastSendTime = -1.;
    float interval = 0;
    int bytesSentPerInterval = 0;
    int bytesReceivedPerInterval = 0;
    int bytesSentPerSecond = 0;
    int bytesReceivedPerSecond = 0;
	bool deleteClient = true;
	bool firstTryUdp = true;
    double time();

};

class VRBSERVEREXPORT VRBClientList
{
protected:
    std::set<VRBSClient *> m_clients;

public:

	VRBSClient *get(covise::Connection *c);
    VRBSClient *get(const char *ip);
    VRBSClient *get(int id);
    VRBSClient *getMaster(const vrb::SessionID &session);
    VRBSClient *getNextInGroup(const vrb::SessionID &id);
    VRBSClient *getNthClient(int N);
    std::vector<VRBSClient *> getClientsWithUserName(const std::string &name);
    int getNextFreeClientID();
    ///client becomes master and all other clients in clients in session lose master state
    void setMaster(VRBSClient *client);
    void setInterval(float i);
    void deleteAll();
    ///send mesage to every member of the session
    void sendMessage(covise::TokenBuffer &stb, const vrb::SessionID &group = vrb::SessionID(0, "all", false), covise::covise_msg_type type = covise::COVISE_MESSAGE_VRB_GUI);
    ///send message to the client with id
    void sendMessageToID(covise::TokenBuffer &stb, int id, covise::covise_msg_type type = covise::COVISE_MESSAGE_VRB_GUI);
    void sendMessageToAll(covise::TokenBuffer &tb, covise::covise_msg_type type = covise::COVISE_MESSAGE_VRB_GUI);
	static std::string cutFileName(const std::string& fileName);
    int numInSession(vrb::SessionID &Group);
    size_t numberOfClients();
    void addClient(VRBSClient *cl);
    void removeClient(VRBSClient *cl);
	/// remove client with connection c
	void remove(covise::Connection *c);
    ///send Message to all clients but the sender of the message
    void passOnMessage(covise::MessageBase* msg, const vrb::SessionID &session = vrb::SessionID(0, "", false));
    ///write the info of all clients in the tokenbuffer
    void collectClientInfo(covise::TokenBuffer &tb);
	VRBSClient* getNextPossibleFileOwner(const std::string& fileName, const vrb::SessionID& id);
};
extern VRBSERVEREXPORT VRBClientList clients;

}
#endif