/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_HOSTLISTHANDLER_H
#define ME_HOSTLISTHANDLER_H

#include <QObject>
#include <QVector>

class QString;
class QColor;
class QAction;

class MEHost;
class MEDaemon;

class MEHostListHandler : public QObject
{
    Q_OBJECT

public:
    MEHostListHandler();
    ~MEHostListHandler();

    static MEHostListHandler *instance();

    int getNoOfHosts()
    {
        return hostList.count();
    };
    int getHostID(const QString &user, const QString &hostname);
    bool isListEmpty()
    {
        return hostList.isEmpty();
    };
    void addHost(MEHost *host);
    void removeHost(MEHost *host);
    void clearList();

    QString getIPAddress(const QString &shortname);
    QString getDNSHostname(const QString &shortname);
    QStringList getList();
    QStringList getList2();
    QVector<MEHost *> getList3();

    MEHost *getFirstHost()
    {
        if (hostList.size() > 0)
            return hostList.first();
        else
            return NULL;
    };
    MEHost *getHostAt(int index)
    {
        return hostList.at(index);
    };
    MEHost *getHost(int hostid);
    MEHost *getHost(const QString &hostname, QString user = "");
    MEHost *getActionHost(QAction *action);

    void addDaemon(MEDaemon *d)
    {
        daemonList.append(d);
    };
    void removeDaemon(const QString &hostname, const QString &user);
    MEDaemon *getDaemon(const QString &hostname, const QString &user);
    QString getDaemon(const QString &shortname);

private:
    QVector<MEHost *> hostList;
    QVector<MEDaemon *> daemonList;
};
#endif
