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
#include "coRegistry.h"

QPixmap *VRBSClient::pix_master = NULL;
QPixmap *VRBSClient::pix_slave = NULL;

VRBClientList clients;

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

int VRBSClient::s_idCounter = 1;
VRBSClient::VRBSClient(Connection *c, const char *ip, const char *n)
{
#ifdef GUI
    if (pix_master == NULL)
    {
        pix_master = new QPixmap(unlockIcon);
        pix_slave = new QPixmap(lockIcon);
    }
#endif
    address = new char[strlen(ip) + 1];
    strcpy(address, ip);
    m_name = new char[strlen(n) + 1];
    strcpy(m_name, n);
    conn = c;
    userInfo = NULL;
    myID = s_idCounter++;
    m_group = -1;
    TokenBuffer rtb;
    rtb << myID;
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

void VRBSClient::setContactInfo(const char *ip, const char *n)
{
    delete[] address;
    delete[] m_name;
    address = new char[strlen(ip) + 1];
    strcpy(address, ip);
    m_name = new char[strlen(n) + 1];
    strcpy(m_name, n);
    TokenBuffer rtb;
    rtb << myID;
    Message m(rtb);
    m.type = COVISE_MESSAGE_VRB_GET_ID;
    conn->send_msg(&m);

// Eintrag anlegen
#ifdef GUI
    char num[100];
    myItem = new QTreeWidgetItem(appwin->table);
    sprintf(num, "%d", myID);
    myItem->setText(ID, num);
    myItem->setText(IP, address);
#endif
}

void VRBSClient::setMaster(int m)
{
    m_master = m;
// Eintrag ndern
#ifdef GUI
    if (m)
        myItem->setIcon(Master, QIcon(*pix_master));
    else
        myItem->setIcon(Master, QIcon(*pix_slave));
#endif
}

void VRBSClient::setGroup(int g)
{
    m_group = g;
#ifdef GUI
    char num[100];
    sprintf(num, "%d", g);
    myItem->setText(Group, num);
#endif
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
    address = new char[strlen("127.0.0.0") + 1];
    strcpy(address, "127.0.0.0");
    m_name = new char[strlen("NONE") + 1];
    strcpy(m_name, "NONE");
    conn = c;
    userInfo = NULL;
    myID = s_idCounter++;
    m_group = -1;
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
    if (coRegistry::instance)
    {
        coRegistry::instance->unObserve(myID);
        coRegistry::instance->deleteEntry(myID);
    }
    delete[] address;
    delete[] m_name;
    delete[] userInfo;
    delete conn;
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
    if (m_name)
    {
        rtb << m_name;
    }
    else
    {
        rtb << "";
    }
    if (userInfo)
    {
        rtb << userInfo;
    }
    else
    {
        rtb << "";
    }
    rtb << m_group;
    rtb << m_master;
}

void VRBSClient::setUserInfo(const char *ui)
{
    delete[] userInfo;
    userInfo = new char[strlen(ui) + 1];
    strcpy(userInfo, ui);

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
    VRBSClient *cl;
    reset();
    while ((cl = current()))
    {
        if (cl->conn == c)
            return cl;
        next();
    }
    return NULL;
}

int VRBClientList::numInGroup(int Group)
{
    VRBSClient *cl;
    int num = 0;
    reset();
    while ((cl = current()))
    {
        if (cl->getGroup() == Group)
        {
            num++;
        }
        next();
    }
    return num;
}

VRBSClient *VRBClientList::get(const char *ip)
{
    VRBSClient *cl;
    reset();
    while ((cl = current()))
    {
        if (cl->getIP() == ip)
            return cl;
        next();
    }
    return NULL;
}

VRBSClient *VRBClientList::get(int id)
{
    VRBSClient *cl;
    reset();
    while ((cl = current()))
    {
        if (cl->getID() == id)
            return cl;
        next();
    }
    return NULL;
}

void VRBClientList::setInterval(float i)
{
    VRBSClient *cl;
    reset();
    while ((cl = current()))
    {
        cl->setInterval(i);
        next();
    }
}

void VRBClientList::deleteAll()
{
    reset();
    while (current())
    {
        remove();
    }
}

void VRBClientList::sendMessage(TokenBuffer &stb, int group, covise_msg_type type)
{
    VRBSClient *cl;
    reset();
    Message m(stb);
    m.type = type;
    while ((cl = current()))
    {
        if ((group == -2) || (group == cl->getGroup()))
        {
            cl->conn->send_msg(&m);
        }
        next();
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
