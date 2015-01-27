/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include "MEHostListHandler.h"
#include "hosts/MEHost.h"
#include "hosts/MEDaemon.h"
#include "widgets/MEUserInterface.h"

/*!
   \class MEHostListHandler
   \brief This class handles the host list
*/

MEHostListHandler::MEHostListHandler()
    : QObject()
{
}

MEHostListHandler *MEHostListHandler::instance()
{
    static MEHostListHandler *singleton = 0;
    if (singleton == 0)
        singleton = new MEHostListHandler();

    return singleton;
}

MEHostListHandler::~MEHostListHandler()
{
    clearList();
}

//!
//! clear list
//!
void MEHostListHandler::clearList()
{
    hostList.clear();
}

//!
//! get the host record for a given hostid
//!
MEHost *MEHostListHandler::getHost(int id)
{
    foreach (MEHost *nptr, hostList)
    {
        if (nptr->getID() == id)
            return (nptr);
    }

    QString buffer = "Host with id " + QString::number(id) + " not found";
    MEUserInterface::instance()->printMessage(buffer);
    return (NULL);
}

//!
//! get a host for a given action
//!
MEHost *MEHostListHandler::getActionHost(QAction *ac)
{
    foreach (MEHost *nptr, hostList)
    {
        if (nptr->getCopyMoveAction() == ac)
            return nptr;
    }
    return NULL;
}

//!
//! get the host record for a given hostname
//!
MEHost *MEHostListHandler::getHost(const QString &name, QString user)
{
    if (!user.isEmpty())
    {
        foreach (MEHost *nptr, hostList)
        {
            if (nptr->getIPAddress() == name && nptr->getUsername() == user)
                return (nptr);
        }
    }

    else
    {
        foreach (MEHost *nptr, hostList)
        {
            if (nptr->getIPAddress() == name)
                return (nptr);
        }
    }

    QString buffer = "Host not found for " + name + " " + user;
    MEUserInterface::instance()->printMessage(buffer);
    return (NULL);
}

//!
//! get the long  hostname (xxx.xxx.xxx.xxx) for a given short hostname
//!
QString MEHostListHandler::getIPAddress(const QString &sname)
{
    foreach (MEHost *nptr, hostList)
    {
        if (nptr->getShortname() == sname)
            return nptr->getIPAddress();
    }
    return (NULL);
}

//!
//! get a host list
//!
QStringList MEHostListHandler::getList()
{
    QStringList list;
    foreach (MEHost *nptr, hostList)
        list << nptr->getText();

    return (list);
}

//!
//! get another host list
//!
QStringList MEHostListHandler::getList2()
{
    QStringList list;

    foreach (MEHost *nptr, hostList)
        list << nptr->getUsername() + "@" + nptr->getShortname();

    return (list);
}

//!
//! get another host list
//!
QVector<MEHost *> MEHostListHandler::getList3()
{
    QVector<MEHost *> list;

    foreach (MEHost *nptr, hostList)
        list << nptr;

    return (list);
}

//!
//! get the hostname + DNS suffix  for a given short hostname
//!
QString MEHostListHandler::getDNSHostname(const QString &sname)
{
    foreach (MEHost *nptr, hostList)
    {
        if (nptr->getShortname() == sname)
            return nptr->getDNSHostname();
    }
    return (NULL);
}

//!
//! add a new host
//!
void MEHostListHandler::addHost(MEHost *nptr)
{
    hostList.append(nptr);
}

//!
//! delete a host, all nodes & depending stuff
//!
void MEHostListHandler::removeHost(MEHost *host)
{
    hostList.remove(hostList.indexOf(host));
    delete host;
}

//!
//! get the host ID for a given hostname
//!
int MEHostListHandler::getHostID(const QString &user, const QString &host)
{
    foreach (MEHost *nptr, hostList)
    {
        if (nptr->getHostname() == host && nptr->getUsername() == user)
            return (nptr->getID());
    }

    return (-1);
}

//!
//! get the host record for a given hostname
//!
void MEHostListHandler::removeDaemon(const QString &hname, const QString &uname)
{
    MEDaemon *daemon = getDaemon(hname, uname);
    if (daemon)
    {
        daemonList.remove(daemonList.indexOf(daemon));
        delete daemon;
    }
}

//!
//! get the host record for a given hostname
//!
MEDaemon *MEHostListHandler::getDaemon(const QString &hname, const QString &uname)
{
    foreach (MEDaemon *nptr, daemonList)
    {
        if (nptr->getHostname() == hname && nptr->getUsername() == uname)
            return (nptr);
    }

    return (NULL);
}

//!
//! get the long  hostname (xxx.xxx.xxx.de) for a given short hostname
//!
QString MEHostListHandler::getDaemon(const QString &sname)
{
    foreach (MEDaemon *nptr, daemonList)
    {
        if (nptr->getShortname() == sname)
            return nptr->getHostname();
    }

    return (NULL);
}
