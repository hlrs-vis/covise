/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef GUI
#include <QSocketNotifier>
#include <QTreeWidget>

#include "gui/VRBapplication.h"

#define IOMANIPH
// don't include iomanip.h becaus it interferes with qt
#endif

#include "VRBClientList.h"
#include <string.h>
#include <net/covise_connect.h>
#include <net/tokenbuffer.h>

#include <iostream>

using namespace covise;
using std::cerr;
using std::endl;
#ifdef GUI
#include "gui/icons/lock.xpm"
#include "gui/icons/unlock.xpm"
#endif

#include <stdlib.h>
#include "VrbServerRegistry.h"

QPixmap *VRBSClient::pix_master = NULL;
QPixmap *VRBSClient::pix_slave = NULL;

VRBClientList clients;
static std::set<int> s_clientIDs;
#ifdef _WIN32

// Windows

#include <sys/timeb.h>
#include <time.h>

double VRBSClient::time()
{
    struct _timeb timebuffer;
    _ftime(&timebuffer);
    return (double)timebuffer.time + 1.e-3 * (double)timebuffer.millitm;
}

#else

// Unix/Linux

#include <sys/time.h>
#include <unistd.h>

double VRBSClient::time()
{
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);

    return (double)tv.tv_sec + 1.e-6 * (double)tv.tv_usec;
}
#endif // _WIN32


VRBSClient::VRBSClient(Connection *c, const char *ip, const char *n)
{
#ifdef GUI
    if (pix_master == NULL)
    {
        pix_master = new QPixmap(unlockIcon);
        pix_slave = new QPixmap(lockIcon);
    }
#endif
    address = ip;
    m_name = n;
    conn = c;
    myID = 1;
    while (s_clientIDs.find(myID) != s_clientIDs.end())
    {
        ++myID;
    }
    s_clientIDs.insert(myID);
    m_group = vrb::SessionID();
    TokenBuffer rtb;
    rtb << myID;
    rtb << m_group;
    Message m(rtb);
    m.type = COVISE_MESSAGE_VRB_GET_ID;
    conn->send_msg(&m);
    m_master = 0;
    socketNotifier = NULL;
    myItem = NULL;
    lastRecTime = 0.0;
    lastSendTime = 0.0;
    for (int j = 0; j < 4; j++)
    {
        myCurves[j] = NULL;
        myLabels[j * 2] = NULL;
        myLabels[j * 2 + 1] = NULL;
    }
}

void VRBSClient::setContactInfo(const char *ip, const char *n, vrb::SessionID &session)
{
    address = ip;
    m_name = n;
    TokenBuffer rtb;
    rtb << myID;
    rtb << session;
    Message m(rtb);
    m.type = COVISE_MESSAGE_VRB_GET_ID;
    conn->send_msg(&m);

// Eintrag anlegen
#ifdef GUI
    char num[100];
    myItem = new QTreeWidgetItem(appwin->table);
    sprintf(num, "%d", myID);
    myItem->setText(ID, num);
    myItem->setText(IP, QString::fromStdString(address));
    setMaster(m_master);
#endif
}

void VRBSClient::setMaster(int m)
{
    m_master = m;
// Eintrag ndern
#ifdef GUI
    if (myItem)
    {
        if (m)
            myItem->setIcon(Master, QIcon(*pix_master));
        else
            myItem->setIcon(Master, QIcon(*pix_slave));
    }
#endif
}

std::string VRBSClient::getUserInfo()
{
    return userInfo;
}

void VRBSClient::setSesion(vrb::SessionID &g)
{
    m_group = g;
#ifdef GUI
    myItem->setText(Group, g.toText().c_str());
#endif
}

int VRBSClient::getMaster()
{
    return m_master;
}

VRBSClient::VRBSClient(Connection *c, QSocketNotifier *sn)
{
#ifdef GUI
    if (pix_master == NULL)
    {
        pix_master = new QPixmap(unlockIcon);
        pix_slave = new QPixmap(lockIcon);
    }
#endif
    socketNotifier = sn;
    address = "localhost";
    m_name = "NONE";
    conn = c;
    myID = 1;
    while (s_clientIDs.find(myID) != s_clientIDs.end())
    {
        ++myID;
    }
    s_clientIDs.insert(myID);
    m_group = vrb::SessionID();
    m_master = 0;
    myItem = NULL;
    interval = 1;
    for (int j = 0; j < 4; j++)
    {
        myCurves[j] = NULL;
        myLabels[j * 2] = NULL;
        myLabels[j * 2 + 1] = NULL;
    }
}

QSocketNotifier *VRBSClient::getSN()
{
    return socketNotifier;
}

VRBSClient::~VRBSClient()
{
    //cerr << "instance" <<coRegistry::instance << endl;
    //cerr << "ID" <<myID << endl;

    delete conn;
    s_clientIDs.erase(myID);
    cerr << "closed connection to client " << myID << endl;
#ifdef GUI
    delete socketNotifier;
    delete myItem;
    appwin->removeCurves(this);
#endif
}

void VRBSClient::setInterval(float i)
{
    interval = i;
}

void VRBSClient::getInfo(TokenBuffer &rtb)
{
    rtb << myID;
    rtb << address;
    rtb << m_name;
    rtb << userInfo;
    rtb << m_group;
    rtb << m_master;
}

void VRBSClient::setUserInfo(const char *ui)
{
    userInfo = ui;

// Eintrag ndern
#ifdef GUI
    char *tmp, *tmp2;
    tmp = new char[strlen(ui) + 1];
    strcpy(tmp, ui);
    char *c = tmp;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    // hostname
    myItem->setText(Host, tmp2);
    c++;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    // userName
    myItem->setText(User, tmp2);
    c++;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    // email
    myItem->setText(Email, tmp2);
    c++;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    // url
    myItem->setText(URL, tmp2);
    c++;
    delete[] tmp;

    appwin->createCurves(this);
#endif
}

std::string VRBSClient::getName() const
{
    return m_name;
}

std::string VRBSClient::getIP() const
{
    return address;
}

int VRBSClient::getID() const
{
    return myID;
}

vrb::SessionID &VRBSClient::getSession()
{
    return m_group;
}

int VRBSClient::getSentBPS()
{
    double currentTime = time();
    if ((currentTime - lastSendTime) > interval)
    {
        bytesSentPerSecond = (int)(bytesSentPerInterval / (currentTime - lastSendTime));
        lastSendTime = currentTime;
        bytesSentPerInterval = 0;
    }
    return bytesSentPerSecond;
}

int VRBSClient::getReceivedBPS()
{
    double currentTime = time();
    if ((currentTime - lastSendTime) > interval)
    {
        bytesReceivedPerSecond = (int)(bytesReceivedPerInterval / (currentTime - lastSendTime));
        lastRecTime = currentTime;
        bytesReceivedPerInterval = 0;
    }
    return bytesReceivedPerSecond;
}

void VRBSClient::addBytesSent(int b)
{
    double currentTime = time();
    bytesSent += b;
    bytesSentPerInterval += b;
    if ((currentTime - lastSendTime) > interval)
    {
        bytesSentPerSecond = (int)(bytesSentPerInterval / (currentTime - lastSendTime));
        lastSendTime = currentTime;
        bytesSentPerInterval = 0;
    }
}

void VRBSClient::addBytesReceived(int b)
{
    double currentTime = time();
    bytesReceived += b;
    bytesReceivedPerInterval += b;
    if ((currentTime - lastSendTime) > interval)
    {
        bytesReceivedPerSecond = (int)(bytesReceivedPerInterval / (currentTime - lastSendTime));
        lastRecTime = currentTime;
        bytesReceivedPerInterval = 0;
    }
}

//
//=========================================================================
//

VRBSClient *VRBClientList::get(Connection *c)
{
    for (VRBSClient *cl : m_clients)
    {
        if (cl->conn == c)
        {
            return cl;
        }
    }
    return NULL;
}

int VRBClientList::numInSession(vrb::SessionID &session)
{
    int num = 0;
    for (auto cl : m_clients)
    {
        if (cl->getSession() == session)
        {
            ++num;
        }
    }
    return num;
}

int VRBClientList::numberOfClients()
{
    return m_clients.size();
}

void VRBClientList::addClient(VRBSClient * cl)
{
    m_clients.insert(cl);
}

void VRBClientList::removeClient(VRBSClient * cl)
{
    m_clients.erase(cl);
    delete cl;
}

void VRBClientList::passOnMessage(covise::Message * msg, const vrb::SessionID &session)
{
    if (session.isPrivate() && session.owner() < 0)
    {
        return;
    }
    for (auto cl : m_clients)
    {
        if (cl->conn != msg->conn && (cl->getSession() == session || session == vrb::SessionID()))
        {
            cl->conn->send_msg(msg);
            cl->addBytesReceived(msg->length);
        }
    }
}

void VRBClientList::collectClientInfo(covise::TokenBuffer & tb)
{
    tb << (int)m_clients.size();
    for (auto cl : m_clients)
    {
        cl->getInfo(tb);
    }
}

VRBSClient *VRBClientList::get(const char *ip)
{
    for (VRBSClient *cl : m_clients)
    {
        if (cl->getIP() == ip)
        {
            return cl;
        }
    }
    return NULL;
}

VRBSClient *VRBClientList::get(int id)
{
    for (VRBSClient *cl : m_clients)
    {
        if (cl->getID() == id)
        {
            return cl;
        }
    }
    return NULL;
}

VRBSClient * VRBClientList::getMaster(const vrb::SessionID &session)
{
    for (auto cl : m_clients)
    {
        if (cl->getSession() == session && cl->getMaster())
        {
            return cl;
        }
    }
    return nullptr;
}

VRBSClient * VRBClientList::getNextInGroup(const vrb::SessionID & id)
{
    for (auto cl : m_clients)
    {
        if (cl->getSession() == id)
        {
            return cl;
        }
    }
    return nullptr;
}

VRBSClient * VRBClientList::getNthClient(int N)
{
    if (N > m_clients.size())
    {
        return nullptr;
    }
    auto it = m_clients.begin();
    std::advance(it, N);
    return *it;
    
}

void VRBClientList::setMaster(VRBSClient * client)
{
    for (auto cl : m_clients)
    {
        if (cl == client)
        {
            cl->setMaster(true);
        }
        else
        {
            cl->setMaster(false);
        }
    }
}

void VRBClientList::setInterval(float i)
{
    for (VRBSClient *cl : m_clients)
    {
        cl->setInterval(i);
    }

}

void VRBClientList::deleteAll()
{
    m_clients.clear();
}

void VRBClientList::sendMessage(TokenBuffer &stb, const vrb::SessionID &group, covise_msg_type type)
{
    Message m(stb);
    m.type = type;

    for (VRBSClient *cl : m_clients)
    {
        if (group == vrb::SessionID(0, std::string(), false) || group == cl->getSession())
        {
            cl->conn->send_msg(&m);
        }
    }
}

void VRBClientList::sendMessageToID(TokenBuffer &stb, int ID, covise_msg_type type)
{
    Message m(stb);
    m.type = type;
    VRBSClient *cl = get(ID);
    if (cl)
    {
        cl->conn->send_msg(&m);
    }
}

void VRBClientList::sendMessageToAll(covise::TokenBuffer &stb, covise::covise_msg_type type)
{
    Message m(stb);
    m.type = type;

    for (VRBSClient *cl : m_clients)
    {
        cl->conn->send_msg(&m);
    }
}
