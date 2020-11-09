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

VrbUiClient::VrbUiClient(Connection *c, UDPConnection* udpc, const char *ip, const char *n, bool send)
    :VRBSClient(c, udpc, ip, n, false)
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
VrbUiClient::VrbUiClient(Connection *c, UDPConnection* udpc, QSocketNotifier *sn)
    :VrbUiClient(c, udpc, "localhost", "NONE", false)
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

void VrbUiClient::setUserInfo(const UserInfo &ui)
{
    VRBSClient::setUserInfo(ui);
    myItem->setText(Host, ui.hostName.c_str());
    myItem->setText(User, ui.ipAdress.c_str());
    myItem->setText(Email, ui.email.c_str());
    myItem->setText(URL, ui.url.c_str());


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
