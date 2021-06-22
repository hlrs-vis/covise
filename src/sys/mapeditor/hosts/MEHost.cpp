/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QMenu>

#include <net/covise_host.h>
#ifdef YAC
#include "yac/coQTSendBuffer.h"
#endif

#include "MEDaemon.h"
#include "MEHost.h"
#include "widgets/MEGraphicsView.h"
#include "widgets/MEUserInterface.h"
#include "widgets/MEModuleTree.h"
#include "handler/MEMainHandler.h"
#include "nodes/MECategory.h"
#include "dataObjects/MEDataTree.h"

;

int MEHost::numHosts = 0; // no. of hosts

/*!
   \class MEHostTreeItem
   \brief This class provides the tree item for the Module List
*/

MEHostTreeItem::MEHostTreeItem(MEHost *h, QTreeWidget *view, int type)
    : QTreeWidgetItem(view, type)
    , m_host(h)
{
}

MEHostTreeItem::~MEHostTreeItem()
{
}

/*!
   \class MEHost
   \brief This class handles the hosts for a Covise session
*/

MEHost::MEHost(int clientId,const std::string &name, const std::string & user)
    : m_clientId(clientId)
    , username(user.c_str())
    , ipname(name.c_str())
{

// get the real hostname for ip address  (visrl.rrz.uni-koeln.de)
    hostname = QString::fromStdString(covise::Host::lookupHostname(name.c_str()));

    // make name for lists
    text = hostname;
    text.prepend("@");
    text.prepend(username);

    init();
}


MEHost::~MEHost()
{
    // make daemon visible if exists
    if (daemon)
        daemon->setVisible(true);

    // delete all modules with this hostname & userid
    // loop over all nodes in network
    if (MEMainHandler::instance())
        MEMainHandler::instance()->removeNodesOfHost(this);

    // delete root of data tree and all entries
    if (dataroot != NULL)
        delete dataroot;

    if (modroot != NULL)
        delete modroot;

    numHosts--;
}

int MEHost::clientId() const
{
    return m_clientId;
}

void MEHost::setClientId(int clientID)
{
    m_clientId = clientID;
}

//!
//! initialize the class
//!
void MEHost::init()
{

    // set hostcolor
    hostcolor = MEMainHandler::instance()->getHostColor(numHosts);

    int hue, s, v;
    hostcolor.getHsv(&hue, &s, &v);
    QColor color_dark;
    color_dark.setHsv(hue, 255, v);

    // make root entry in data tree
    dataroot = new MEDataTreeItem(MEDataTree::instance(), text, hostcolor);
    QBrush brush(color_dark, Qt::SolidPattern);
    dataroot->setForeground(0, brush);
    dataroot->setExpanded(true);

    // make root item for module browser list
    modroot = new MEHostTreeItem(this, MEUserInterface::instance()->getModuleTree());
    modroot->setText(0, text);
    modroot->setForeground(0, brush);
    modroot->setExpanded(numHosts == 0 ? true : false);

    numHosts++;

    // set an action for this host
    // used by node menu for move to/copy to
    m_categoryMenu = new QMenu(0);
    m_hostAction = new QAction(username + "@" + hostname, 0);
    m_copyMoveAction = new QAction(username + "@" + hostname, 0);
    QObject::connect(m_hostAction, SIGNAL(hovered()), MEGraphicsView::instance(), SLOT(hoveredHostCB()));
    QObject::connect(m_copyMoveAction, SIGNAL(triggered()), MEGraphicsView::instance(), SLOT(triggeredHostCB()));
    m_hostAction->setMenu(m_categoryMenu);

    QVariant var;
    var.setValue(this);
    m_hostAction->setData(var);
    m_copyMoveAction->setData(var);
}

//!
//!  add categories and modulenames coming from controller
//!
void MEHost::addHostItems(const std::vector<std::string> &modules, const std::vector<std::string> &categories)
{
    assert(modules.size() == categories.size());
    for (size_t i = 0; i < modules.size(); i++)
    {
        if (categories[i] != "SRenderer")
        {
            MECategory *cptr = getCategory(categories[i].c_str());
            cptr->addModuleName(modules[i].c_str());
        }
    }
}

//!
//!  set the right icon
//!
void MEHost::setGUI(bool state)
{
    gui = state;
    if (state)
        m_icon = MEMainHandler::instance()->pm_adduser;
    else
        m_icon = MEMainHandler::instance()->pm_addhost;

    modroot->setIcon(0, m_icon);
}


//!
//!  get the category pointer for a certain name on a host
//!
MECategory *MEHost::getCategory(const QString &category)
{
    foreach (MECategory *cptr, catList)
    {
        if (cptr->getName() == category)
            return cptr;
    }

    MECategory *tmp_ptr = new MECategory(category);
    catList << tmp_ptr;
    m_categoryMenu->addAction(tmp_ptr->getAction());

    return tmp_ptr;
}
