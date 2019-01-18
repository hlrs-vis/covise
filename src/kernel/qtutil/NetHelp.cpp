/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "NetHelp.h"
#include <iostream>
#include <QHostInfo>
#include <net/covise_host.h>

namespace covise
{

QString NetHelp::getLocalIP()
{
    Host host;
    if (host.getName())
        return GetIpAddress(host.getName());

    return QString("unknown");
}

QString NetHelp::GetIpAddress(const char *hostname)
{
    QHostInfo host = QHostInfo::fromName(QString(hostname));
    QList<QHostAddress> list = host.addresses();
    QHostAddress address = list.at(0);
    return address.toString();
}

QString NetHelp::GetNamefromAddr(QString *address)
{
    QHostInfo host = QHostInfo::fromName(*address);
    QString hostName = host.hostName();
    return hostName;
}

}
