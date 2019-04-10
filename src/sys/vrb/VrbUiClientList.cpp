#include "VrbUiClientList.h"
/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <qtutil/NetHelp.h>
#include <qtutil/FileSysAccess.h>
#include <QtCore/qfileinfo.h>
#include <QtCore/qdir.h>
#include <QtNetwork/qhostinfo.h>
#include <QSocketNotifier>
#include <QTreeWidget>

#include "gui/VRBapplication.h"
#include "gui/icons/lock.xpm"
#include "gui/icons/unlock.xpm"

using namespace covise;
using namespace vrb;
VrbUiClientList uiClients;

QPixmap *VrbUiClient::pix_master;
QPixmap *VrbUiClient::pix_slave;

VrbUiClient::VrbUiClient(Connection *c, const char *ip, const char *n)
    :VRBSClient(c,ip, n)
{
    if (pix_master == NULL)
    {
        pix_master = new QPixmap(unlockIcon);
        pix_slave = new QPixmap(lockIcon);
    }
    socketNotifier = NULL;
    myItem = NULL;
    for (int j = 0; j < 4; j++)
    {
        myCurves[j] = NULL;
        myLabels[j * 2] = NULL;
        myLabels[j * 2 + 1] = NULL;
    }
}
VrbUiClient::VrbUiClient(Connection *c, QSocketNotifier *sn)
    :VrbUiClient(c, "localhost", "NONE")
{
    socketNotifier = sn;
}

VrbUiClient::~VrbUiClient()
{
    delete socketNotifier;
    delete myItem;
    appwin->removeCurves(this);
}


void VrbUiClient::setContactInfo(const char * ip, const char * n, SessionID & session)
{
    VRBSClient::setContactInfo(ip, n, session);

    char num[100];
    myItem = new QTreeWidgetItem(appwin->table);
    sprintf(num, "%d", myID);
    myItem->setText(ID, num);
    myItem->setText(IP, ip);
    setMaster(m_master);
}
void VrbUiClient::setMaster(bool m)
{
    VRBSClient::setMaster(m);
    if (myItem)
    {
        if (m)
            myItem->setIcon(Master, QIcon(*pix_master));
        else
            myItem->setIcon(Master, QIcon(*pix_slave));
    }
}
void VrbUiClient::setSession(const SessionID &id)
{
    VRBSClient::setSession(id);
    myItem->setText(Group, id.toText().c_str());
}

void VrbUiClient::setUserInfo(const char *ui)
{
    VRBSClient::setUserInfo(ui);
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
}

QSocketNotifier * VrbUiClient::getSN()
{
    return socketNotifier;
}
///////////////////////////////////////////////////////////////
VrbUiClient * VrbUiClientList::get(QSocketNotifier * sn)
{
    for (VRBSClient *cl : m_clients)
    {
        VrbUiClient *uicl = static_cast<VrbUiClient *>(cl);
        if (uicl->getSN() == sn)
        {
            return uicl;
        }
    }
    return NULL;
}
