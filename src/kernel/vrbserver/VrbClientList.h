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
namespace covise
{
class TokenBuffer;
class Connection;
class Message;
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
class VRBSERVEREXPORT VRBSClient
{
public:
    ///Vrb Server client that holds a connection and information about the client
    ///send = false if you dont want to inform the client immediatly about the contact
    VRBSClient(covise::Connection *c, const char *ip, const char *name , bool send = true);
    virtual ~VRBSClient();
    ///set clientinformation and inform the client about its server id and session
    virtual void setContactInfo(const char *ip, const char *n, vrb::SessionID &session);
    ///store userinfo like email, pc-name, ...
    virtual void setUserInfo(const char *userInfo);
    covise::Connection *conn;
    std::string getName() const;
    std::string getIP() const;
    int getID() const;
    vrb::SessionID &getSession();
    virtual void setSession(const vrb::SessionID &g);
    vrb::SessionID &getPrivateSession();
    void setPrivateSession(vrb::SessionID &g);
    int getMaster();
    virtual void setMaster(bool m);
    std::string getUserInfo();
    std::string getUserName();
    int getSentBPS();
    int getReceivedBPS();
    void setInterval(float i);
    void addBytesSent(int b);
    void addBytesReceived(int b);
    void getInfo(covise::TokenBuffer &rtb);


protected:
    std::string address;
    std::string m_name;
    std::string userInfo;
    int myID = -1;
    vrb::SessionID m_publicSession, m_privateSession;
    int m_master = false;
    long bytesSent;
    long bytesReceived;
    double lastRecTime = -1.;
    double lastSendTime = -1.;
    float interval;
    int bytesSentPerInterval;
    int bytesReceivedPerInterval;
    int bytesSentPerSecond;
    int bytesReceivedPerSecond;
    
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
    int numInSession(vrb::SessionID &Group);
    int numberOfClients();
    void addClient(VRBSClient *cl);
    void removeClient(VRBSClient *cl);
    ///send Message to all clients but the sender of the message
    void passOnMessage(covise::Message * msg, const vrb::SessionID &session = vrb::SessionID(0, "", false));
    ///write the info of all clients in the tokenbuffer
    void collectClientInfo(covise::TokenBuffer &tb);
};
extern VRBSERVEREXPORT VRBClientList clients;

}
#endif