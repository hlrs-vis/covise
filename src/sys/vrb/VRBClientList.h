/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRB_CLIENT_LIST_H
#define _VRB_CLIENT_LIST_H
#include <util/DLinkList.h>
#include <net/tokenbuffer.h>
#include <net/message_types.h>

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

    void setContactInfo(const char *ip, const char *n);
    void setUserInfo(const char *userInfo);
    covise::Connection *conn;
    const char *getName()
    {
        return m_name;
    };
    const char *getIP()
    {
        return address;
    };
    int getID()
    {
        return myID;
    };
    int getGroup()
    {
        return m_group;
    };
    void setGroup(int g);
    int getMaster()
    {
        return m_master;
    };
    void setMaster(int m);
    const char *getUserInfo()
    {
        return userInfo;
    };
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
    char *address = nullptr;
    char *m_name = nullptr;
    char *userInfo = nullptr;
    int myID = -1;
    int m_group = -1;
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

class VRBClientList : public covise::DLinkList<VRBSClient *>
{

public:
    VRBSClient *get(QSocketNotifier *c);
    VRBSClient *get(covise::Connection *c);
    VRBSClient *get(const char *ip);
    VRBSClient *get(int id);
    void setInterval(float i);
    void deleteAll();
    void sendMessage(covise::TokenBuffer &stb, int group = -2, covise::covise_msg_type type = covise::COVISE_MESSAGE_VRB_GUI);
    void sendMessageToID(covise::TokenBuffer &stb, int id, covise::covise_msg_type type = covise::COVISE_MESSAGE_VRB_GUI);
    int numInGroup(int Group);
};

extern VRBClientList clients;
#endif
