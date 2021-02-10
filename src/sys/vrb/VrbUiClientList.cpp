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
#include <sstream>
#include "gui/VRBapplication.h"
#include "gui/icons/lock.xpm"
#include "gui/icons/unlock.xpm"

using namespace covise;
using namespace vrb;
VrbUiClientList uiClients;

QPixmap *VrbUiClient::pix_master;
QPixmap *VrbUiClient::pix_slave;

VrbUiClient::VrbUiClient(const Connection *c, const UDPConnection* udpc, QSocketNotifier *sn, covise::TokenBuffer& tb)
    : VRBSClient(c, udpc, tb)
    , socketNotifier(sn)
{
    if (pix_master == nullptr)
    {
        pix_master = new QPixmap(unlockIcon);
        pix_slave = new QPixmap(lockIcon);
    }
    myItem = nullptr;
    for (int j = 0; j < 4; j++)
    {
        myCurves[j] = nullptr;
        myLabels[j * 2] = nullptr;
        myLabels[j * 2 + 1] = nullptr;
    }

    myItem = new QTreeWidgetItem(appwin->table);
    myItem->setText(vrb::Columns::ID, QString::number(ID()));
    myItem->setText(IP, userInfo().ipAdress.c_str());
    setMaster(isMaster());

    myItem->setText(vrb::Columns::Host, userInfo().hostName.c_str());
    myItem->setText(User, userInfo().ipAdress.c_str());
    myItem->setText(Email, userInfo().email.c_str());
    myItem->setText(URL, userInfo().url.c_str());

    appwin->createCurves(this);
}

VrbUiClient::~VrbUiClient()
{
    delete socketNotifier;
    delete myItem;
    appwin->removeCurves(this);
}

void VrbUiClient::setMaster(int clientID)
{
    VRBSClient::setMaster(clientID);
    if (myItem)
    {
        if (isMaster())
            myItem->setIcon(Master, QIcon(*pix_master));
        else
            myItem->setIcon(Master, QIcon(*pix_slave));
    }
    std::stringstream ss;
    ss << sessionID();
    myItem->setText(Group, ss.str().c_str());
}
void VrbUiClient::setSession(const SessionID &id)
{
    VRBSClient::setSession(id);
    std::stringstream ss;
    ss << id;
    myItem->setText(Group, ss.str().c_str());
}


QSocketNotifier * VrbUiClient::getSN()
{
    return socketNotifier;
}
///////////////////////////////////////////////////////////////
VrbUiClient * VrbUiClientList::get(QSocketNotifier * sn)
{
    for (auto &cl : m_clients)
    {
        VrbUiClient *uicl = dynamic_cast<VrbUiClient *>(cl.get());
        if (uicl->getSN() == sn)
        {
            return uicl;
        }
    }
    return nullptr;
}
