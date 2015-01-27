/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_HOST_H
#define ME_HOST_H

#include <QVector>
#include <QTreeWidgetItem>

namespace covise
{
class coRecvBuffer;
}

class QAction;
class QColor;
class QString;

class MEHost;
class MEDataTreeItem;
class MEDaemon;
class MECategory;

//================================================
class MEHostTreeItem : public QTreeWidgetItem
//================================================
{
public:
    explicit MEHostTreeItem(MEHost *h, QTreeWidget *view, int type = Type);
    ~MEHostTreeItem();

    MEHost *getHost()
    {
        return m_host;
    };

private:
    MEHost *m_host;
};

//================================================
class MEHost
//================================================
{

public:
    MEHost();
    MEHost(int hostno, QString hostname, int noOfNodes, QString user);
    MEHost(QString name, QString user);
    ~MEHost();

    QVector<MECategory *> catList; // list of categories
    QVector<MEHost *> mirrorList; // list of hosts for mirroring

    bool hasGUI()
    {
        return gui;
    };
    int getID()
    {
        return hostid;
    };
    int getNumNo()
    {
        return numNo;
    };
    void addHostItems(const QStringList &);
    void setGUI(bool state);
    void setDaemon(MEDaemon *d)
    {
        daemon = d;
    };
    void addHostItems(covise::coRecvBuffer &);
    void setIcon(int mode);
    QPixmap getIcon()
    {
        return m_icon;
    };

    QAction *getHostAction()
    {
        return m_hostAction;
    };
    QAction *getCopyMoveAction()
    {
        return m_copyMoveAction;
    };
    QMenu *getMenu()
    {
        return m_categoryMenu;
    };
    QColor getColor()
    {
        return hostcolor;
    };
    QString getUsername()
    {
        return username;
    };
    QString getShortname()
    {
        return shortname;
    };
    QString getDNSHostname()
    {
        return hostname;
    }
    QString getText()
    {
        return text;
    }
    QTreeWidgetItem *getModuleRoot()
    {
        return modroot;
    };
    QString getIPAddress()
    {
        return ipname;
    };
    QString getHostname()
    {
        return hostname;
    };

    MEDaemon *getDaemon()
    {
        return daemon;
    };
    MEDataTreeItem *getDataRoot()
    {
        return dataroot;
    };
    MECategory *getCategory(const QString &);

    QVector<MEHost *> mirrorNames;

private:
    static int numHosts; // no. of hosts
    bool gui; // host has an user interface
    int hostid;
    int numNo; // no. of parallel nodes, normally 1

    void init();

    MEDataTreeItem *dataroot;
    MEDaemon *daemon; // daemon of host

    QAction *m_hostAction, *m_copyMoveAction;
    QMenu *m_categoryMenu;
    QPixmap m_icon;
    QString text; // text in lists
    QString hostname; // full hostname
    QString shortname; // short hostname
    QString username; // user id, not used
    QString ipname; // ip address
    QColor hostcolor; // current color for host
    MEHostTreeItem *modroot;
};

Q_DECLARE_METATYPE(MEHost *);
#endif
