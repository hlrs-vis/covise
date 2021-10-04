/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_HOST_H
#define ME_HOST_H

#include <QVector>
#include <QTreeWidgetItem>
#include <net/userinfo.h>

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
    MEHost(int clientId, const covise::UserInfo &hostInfo);
    ~MEHost();
    int clientId() const;
    void setClientId(int clientID);
    QVector<MECategory *> catList; // list of categories
    QVector<MEHost *> mirrorList;  // list of hosts for mirroring

    bool hasGUI()
    {
        return gui;
    };
    int getNumNo()
    {
        return numNo;
    };
    void addHostItems(const std::vector<std::string> &modules, const std::vector<std::string> &categories);
    void setGUI(bool state);
    void setDaemon(MEDaemon *d)
    {
        daemon = d;
    };
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
    QString getUsername();
    QString getDNSHostname();
    QString getText();
    QTreeWidgetItem *getModuleRoot();
    QString getIPAddress();
    QString getHostname();

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
    bool gui = false;    // host has an user interface
    int numNo = -1;      // no. of parallel nodes, normally 1

    void init();

    MEDataTreeItem *dataroot;
    MEDaemon *daemon = nullptr; // daemon of host

    QAction *m_hostAction, *m_copyMoveAction;
    QMenu *m_categoryMenu;
    QPixmap m_icon;
    const covise::UserInfo m_hostInfo;
    QColor hostcolor; // current color for host
    MEHostTreeItem *modroot;
    int m_clientId = 0;
};

Q_DECLARE_METATYPE(MEHost *);
#endif
