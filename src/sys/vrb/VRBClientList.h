/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRB_CLIENT_LIST_H
#define _VRB_CLIENT_LIST_H
#include <util/DLinkList.h>
#include <net/tokenbuffer.h>
#include <net/message_types.h>
#include <vrbclient/SessionID.h>
#include <set>
namespace covise
{
class ServerConnection;
class Connection;
class ConnectionList;
class Message;
class TokenBuffer;
}

class QSocketNotifier;
class QTreeWidgetItem;
class QPixmap;
class QLabel;
class VRBCurve;

class VRBSClient
{
public:
    VRBSClient(covise::Connection *c, QSocketNotifier *sn);
    VRBSClient(covise::Connection *c, const char *ip, const char *m_name);
    ~VRBSClient();

    enum Columns {
        Master,
        ID,
        Group,
        User,
        Host,
        Email,
        URL,
        IP,
    };

    void setContactInfo(const char *ip, const char *n, vrb::SessionID &session);
    void setUserInfo(const char *userInfo);
    covise::Connection *conn;
    std::string getName() const;
    std::string getIP() const;
    int getID() const;
    vrb::SessionID &getSession();
    void setSesion(vrb::SessionID &g);
    int getMaster();
    void setMaster(int m);
    std::string getUserInfo();
    int getSentBPS();
    int getReceivedBPS();
    void setInterval(float i);
    void addBytesSent(int b);
    void addBytesReceived(int b);
    void getInfo(covise::TokenBuffer &rtb);
    QSocketNotifier *getSN();

    QTreeWidgetItem *myItem;
    VRBCurve *myCurves[4];
    QLabel *myLabels[8];
    static QPixmap *pix_master;
    static QPixmap *pix_slave;

private:
    std::string address;
    std::string m_name;
    std::string userInfo;
    int myID = -1;
    vrb::SessionID m_group;
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
    QSocketNotifier *socketNotifier;
    static int s_idCounter;
    double time();
};

class VRBClientList 
{
private:
    std::set<VRBSClient *> m_clients;
    
 
public:
    VRBSClient *get(QSocketNotifier *c);
    VRBSClient *get(covise::Connection *c);
    VRBSClient *get(const char *ip);
    VRBSClient *get(int id);
    VRBSClient *getMaster(const vrb::SessionID &session);
    VRBSClient *getNextInGroup(const vrb::SessionID &id);
    VRBSClient *getNthClient(int N);
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
    void passOnMessage(covise::Message * msg, const vrb::SessionID &session = vrb::SessionID());
    ///write the info of all clients in the tokenbuffer
    void collectClientInfo(covise::TokenBuffer &tb);
};

extern VRBClientList clients;
#endif
