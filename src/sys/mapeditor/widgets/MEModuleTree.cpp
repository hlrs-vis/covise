/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QMenu>
#include <QMimeData>
#include <QContextMenuEvent>
#include <QDebug>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QSslSocket>
#include <QApplication>
#include <QTime>

#include <covise/covise_msg.h>
#include <covise/covise_version.h>

#include "MEModuleTree.h"
#include "MEUserInterface.h"
#include "MEMessageHandler.h"
#include "nodes/MECategory.h"
#include "hosts/MEHost.h"
#include "handler/MEFavoriteListHandler.h"
#include "handler/MEMainHandler.h"
#include "handler/MEHostListHandler.h"
#include "widgets/MEGraphicsView.h"
#include "util/unixcompat.h"

;

MEMainHandler *MEModuleTree::m_mainHandler = NULL;
QMap<QString, QString> MEModuleTree::s_moduleHelp;

/*!
    \class MEModuleTreeItem
    \brief Class sorts module tree items (case sensitive)
*/

class MEModuleTreeItem : public QTreeWidgetItem
{
public:
    MEModuleTreeItem(int type = Type)
        : QTreeWidgetItem(type)
    {
    }
    MEModuleTreeItem(QTreeWidget *widget, int type = Type)
        : QTreeWidgetItem(widget, type)
    {
    }
    MEModuleTreeItem(QTreeWidgetItem *parent, int type = Type)
        : QTreeWidgetItem(parent, type)
    {
    }
    virtual ~MEModuleTreeItem() {}

    virtual bool operator<(const QTreeWidgetItem &other) const
    {
        QString myData = data(0, Qt::DisplayRole).toString();
        QString otherData = other.data(0, Qt::DisplayRole).toString();
        return myData.compare(otherData, Qt::CaseInsensitive) < 0;
    }
};

/*!
    \class MEModuleTree
    \brief Tree widget shows and handles a list of hosts, categories and modules

    This widget is part of the MEUserInterface.
*/

MEModuleTree::MEModuleTree(QWidget *parent)
    : QTreeWidget(parent)
{
    const char *text4 = "<h4>Module Browser</h4><h4>Overview</h4>"
                        "<p>This area contains a hierarchy (tree) displaying the hostnames, category names and module names.</p>"
                        "<p> When started in single user mode only the name of the local host is shown in the tree. "
                        "Modules running on a host appear in the same color as the host name in the list. "

                        "<h4>Tooltips</h4>"
                        "<p>Short description of the category and the modules are shown as tooltips. "
                        "Clicking on a host with the right mouse button opens a popup menu with one single entry <em>Delete Host</em>. "
                        "Clicking on this item with the left mouse button will remove all modules running on that host "
                        "and a possible remote user interface. "
                        "Clicking on a category or module with the right mouse button will open the help system. </p>"

                        "<h4>Using</h4>"
                        "The modules shown for each category depend on the chosen modules in former sessions. To simplify the view "
                        "only modules which have been used before are shown. All other modules are hidden "
                        "behind the item <em>More...</em>. Clicking on this item will show all available modules in this category."

                        "The special category <em>All</em> contains all available modules in alphabetic order and the corresponding category."

                        "<ul>"
                        "<li>Clicking on the +/- sign open/close the corresponding category. </li>"
                        "<li>Double clicking on a category name opens this specific category and closes all others.</li>"
                        "<li>Clicking into the <em>canvas</em> and typing / allows to enter a search string. All categories containing this string in the "
                        "module name will be highlighted. Icons of these module in the <em>canvas</em> will also be highlighted.</li>"
                        "</ul>"
                        "To start a module its module name has to be dragged to the canvas.  "
                        "An <em>icon</em> on the canvas indicates, that the respective  program representing the module has "
                        "been started and waits for its execution.";

    m_mainHandler = MEMainHandler::instance();

    // init tree widget
    setColumnCount(1);
    setHeaderLabel("Module List");
    resizeColumnToContents(0);
    setSelectionMode(QAbstractItemView::SingleSelection);
    setSelectionBehavior(QAbstractItemView::SelectItems);
    setDragEnabled(true);
    setDragDropMode(QAbstractItemView::DragDrop);
    setContextMenuPolicy(Qt::DefaultContextMenu);
    setAcceptDrops(true);
    setDropIndicatorShown(true);

    connect(this, SIGNAL(itemDoubleClicked(QTreeWidgetItem *, int)), this, SLOT(doubleClicked(QTreeWidgetItem *, int)));
    connect(this, SIGNAL(itemClicked(QTreeWidgetItem *, int)), this, SLOT(clicked(QTreeWidgetItem *, int)));
    connect(this, SIGNAL(itemCollapsed(QTreeWidgetItem *)), this, SLOT(collapsed(QTreeWidgetItem *)));
    connect(this, SIGNAL(itemExpanded(QTreeWidgetItem *)), this, SLOT(expanded(QTreeWidgetItem *)));

    setWhatsThis(text4);

    // create popup menus for right mouse button
    // hosts
    m_hostMenu = new QMenu(this);

    m_delhost_a = m_hostMenu->addAction("Delete Host");
    m_delhost_a->setToolTip("Remove this host from the session");
    connect(m_delhost_a, SIGNAL(triggered()), this, SLOT(removeHostCB()));
    m_hostMenu->addSeparator();

    // module & category help --> opens COVISE help
    m_categoryMenu = new QMenu(this);
    m_help_a = m_categoryMenu->addAction("Help...");
    m_help_a->setToolTip("Open help system");
    connect(m_help_a, SIGNAL(triggered()), this, SLOT(infoCB()));

    // module & category help --> opens COVISE help
    m_moduleMenu = new QMenu(this);
    m_help_a = m_moduleMenu->addAction("Help...");
    m_help_a->setToolTip("Open help system");
    m_separator_a = m_moduleMenu->addSeparator();
    m_exec_debug_a = m_moduleMenu->addAction("Start in Debugger");
    m_exec_memcheck_a = m_moduleMenu->addAction("Check for Memory Errors");
    developerMode(m_mainHandler->cfg_DeveloperMode);
    connect(MEMainHandler::instance(), SIGNAL(developerMode(bool)), this, SLOT(developerMode(bool)));
    connect(m_help_a, SIGNAL(triggered()), this, SLOT(infoCB()));
    connect(m_exec_debug_a, SIGNAL(triggered()), this, SLOT(execDebugCB()));
    connect(m_exec_memcheck_a, SIGNAL(triggered()), this, SLOT(execMemcheckCB()));
}

MEModuleTree::~MEModuleTree()
{
}

void MEModuleTree::readModuleTooltips()
{
    // create a dictionary for category description
    if (!s_moduleHelp.isEmpty())
        return;

    s_moduleHelp.insert("All", "All modules sorted alphabetically");
    s_moduleHelp.insert("Color", "Transform abstract data to color");
    s_moduleHelp.insert("Converter", "Convert to a different representation");
    s_moduleHelp.insert("Examples", "Examples for COVISE programmers");
    s_moduleHelp.insert("Filter", "Select data from a larger amount");
    s_moduleHelp.insert("Interpolator", "Interpolate data to different spatial and temporal positions");
    s_moduleHelp.insert("HLRS", "Modules for HLRS users");
    s_moduleHelp.insert("IO", "Read or write data in a variety of formats");
    s_moduleHelp.insert("Mapper", "Generate geometric representations of abstract data");
    s_moduleHelp.insert("Obsolete", "Unmaintained modules superseded by others");
    s_moduleHelp.insert("Reader", "Read data in a variety of formats");
    s_moduleHelp.insert("Renderer", "Display geometric data on the desktop or in VR");
    s_moduleHelp.insert("Simulation", "Online coupling and steering of simulations");
    s_moduleHelp.insert("Test", "Test modules");
    s_moduleHelp.insert("Tools", "Helpful tools that do not fit into any of the other categories");
    s_moduleHelp.insert("Tracer", "Generate particle traces in vector fields");
    s_moduleHelp.insert("UnderDev", "Still under development");
    s_moduleHelp.insert("VISiT", "Virtual Intuitive Simulation Testbed (IHS Stuttgart)");
    s_moduleHelp.insert("VTK", "Visualization Toolkit (VTK) classes");

    QString basedir = getenv("COVISEDIR");
    QFile tipsfile(basedir + "/doc/moduledescriptions.txt");
    QNetworkRequest request;
    QNetworkAccessManager *manager = new QNetworkAccessManager(this);
    ;
    connect(manager,
            SIGNAL(sslErrors(QNetworkReply *, const QList<QSslError> &)),
            MEMainHandler::instance(),
            SLOT(handleSslErrors(QNetworkReply *, const QList<QSslError> &)));
    QIODevice *iodev = NULL;
    if (tipsfile.open(QFile::ReadOnly))
    {
        iodev = &tipsfile;
    }
    else
    {
        request.setUrl(QUrl("https://fs.hlrs.de/projects/covise/doc/moduledescriptions.txt"));
        char ua[1024];
        snprintf(ua, sizeof(ua), "COVISE Map Editor (%s)", covise::CoviseVersion::shortVersion());
        request.setRawHeader("User-Agent", ua);

        iodev = manager->get(request);
        OperationWaiter waiter;
        connect(iodev, SIGNAL(finished()), &waiter, SLOT(finished()));
        waiter.wait(2000); // wait at most 2 seconds
    }

    if (iodev)
    {
        QTextStream stream(iodev);
        while (!stream.atEnd())
        {
            QString line = stream.readLine();
            QString catmod = line.section(":", 0, 0);
            QString desc = line.section(":", 1).simplified();
            s_moduleHelp.insert(catmod, desc);
        }
        iodev->close();
    }
}

void MEModuleTree::developerMode(bool devmode)
{
    m_separator_a->setVisible(devmode);
    m_exec_debug_a->setVisible(devmode);
    m_exec_memcheck_a->setVisible(devmode);
}

void MEModuleTree::hideUnusedItems(QTreeWidgetItem *category)
{
    if (!category)
        return;

    if (getDepth(category) != 1)
        return; // not a category

    bool hide = m_mainHandler->cfg_HideUnusedModules;

    int numVisible = 0;
    QString cname = category->data(0, Qt::DisplayRole).toString();
    for (int i = 0; i < category->childCount(); ++i)
    {
        QTreeWidgetItem *module = category->child(i);
        QString mname = module->data(0, Qt::DisplayRole).toString();

        bool visible = false;
        if (cname == "All")
            visible = m_usedModules.contains(mname);
        else
            visible = m_usedModules.contains(mname + " (" + cname + ")");

        if (visible)
            ++numVisible;
    }

    bool showAll = !hide;
    if (numVisible == 0 || numVisible >= category->childCount() - 2) // show everything if using "More..." would not reduce the number of items
        showAll = true;

    for (int i = 0; i < category->childCount(); ++i)
    {
        QTreeWidgetItem *module = category->child(i);
        QString mname = module->data(0, Qt::DisplayRole).toString();

        if (mname == "More...")
        {
            module->setHidden(showAll || category->childCount() == numVisible + 1);
            continue;
        }

        bool visible = false;
        if (cname == "All")
            visible = m_usedModules.contains(mname);
        else
            visible = m_usedModules.contains(mname + " (" + cname + ")");

        if (visible && hide)
            module->setFont(0, MEMainHandler::s_boldFont);
        else
            module->setFont(0, MEMainHandler::s_normalFont);
        if (showAll)
            module->setHidden(false);
        else
            module->setHidden(!visible);
    }
}

//!
//! Add the module list received for a certain host, populate the tree
//!
void MEModuleTree::addHostList(MEHost *currentHost)
{
    readModuleTooltips();

    //bool hideItems = false;
    // hide module item if something is in the history for former used modules
    //if(!m_mainHandler->moduleHistory.isEmpty())
    //   hideItems = true;

    // get module history from mapeditor.xml
    // do this only for beginner mode
    if (m_mainHandler->cfg_HideUnusedModules)
        highlightHistoryModules();

    // get root item of module browser (normally the hostname)
    QTreeWidgetItem *rootItem = currentHost->getModuleRoot();

    // create the new category ALL
    // make root & the "More... "item for this category
    MECategory *categoryAll = new MECategory("All", "All");
    currentHost->catList.append(categoryAll);

    MEModuleTreeItem *allItem = new MEModuleTreeItem(rootItem);
    allItem->setText(0, "All");
    allItem->setToolTip(0, s_moduleHelp.value("All"));
    allItem->setIcon(0, m_mainHandler->pm_folderclosed);

    MEModuleTreeItem *allMoreItem = new MEModuleTreeItem();
    allMoreItem->setText(0, "More...");
    allMoreItem->setIcon(0, m_mainHandler->pm_bulletmore);
    allMoreItem->setFont(0, MEMainHandler::s_italicFont);
    allMoreItem->setToolTip(0, "Show all modules");
    allMoreItem->setFlags(allMoreItem->flags() & ~Qt::ItemIsDragEnabled);

    categoryAll->setCategoryItem(allItem);

    // insert modules and categories
    // categories have already been created in MEHost constructor
    foreach (MECategory *category, currentHost->catList)
    {
        QString name = category->getName();

        if (name == "All")
            continue;

        MEModuleTreeItem *categoryItem = new MEModuleTreeItem(rootItem);
        categoryItem->setToolTip(0, s_moduleHelp.value(name));
        categoryItem->setText(0, name);
        categoryItem->setIcon(0, m_mainHandler->pm_folderclosed);
        category->setCategoryItem(categoryItem);

        // generate item "More..." for this category
        MEModuleTreeItem *moreItem = new MEModuleTreeItem();
        moreItem->setText(0, "More...");
        moreItem->setIcon(0, m_mainHandler->pm_bulletmore);
        moreItem->setFont(0, MEMainHandler::s_italicFont);
        moreItem->setToolTip(0, "Show all modules of this category");
        moreItem->setFlags(moreItem->flags() & ~Qt::ItemIsDragEnabled);

        // insert all module names to category and ALL
        int nc = category->moduleList.count();
        for (int i = 0; i < nc; i++)
        {
            QString mname = category->moduleList.at(i);

            QString tooltip = s_moduleHelp.value(name + "/" + mname);

            MEModuleTreeItem *moduleItem = new MEModuleTreeItem(categoryItem);
            moduleItem->setText(0, mname);
            moduleItem->setIcon(0, m_mainHandler->pm_bullet);
            moduleItem->setToolTip(0, tooltip);

            MEModuleTreeItem *allModuleItem = new MEModuleTreeItem(allItem);
            allModuleItem->setText(0, mname + " (" + name + ")");
            allModuleItem->setIcon(0, m_mainHandler->pm_bullet);
            allModuleItem->setToolTip(0, tooltip);
        }

        // sort list and add "More..." item
        categoryItem->sortChildren(0, Qt::AscendingOrder);
        categoryItem->insertChild(0, moreItem);
    }

    // sort all items in category "ALL"
    // this list can't be sorted correctly ( perhaps to many items? )
    // it will be sorted if all items have the same hidden state
    allItem->sortChildren(0, Qt::AscendingOrder);
    allItem->insertChild(0, allMoreItem);

    // look in mapeditor.xml for already opened category folder
    // this must be the last step otherwise the hidden items were not shown correctly
    foreach (MECategory *category, currentHost->catList)
    {
        bool ocstate = m_mainHandler->getConfig()->isOn(category->getName(), "System.MapEditor.General.Category", false);
        if (ocstate)
        {
            QTreeWidgetItem *item = category->getCategoryItem();
            item->setExpanded(true);
            item->setIcon(0, m_mainHandler->pm_folderopen);
        }
    }
    allItem->setExpanded(false);
    allItem->setIcon(0, m_mainHandler->pm_folderclosed);
}

//!
//! Get modules from module history (mapeditor.xml) when beginner mode is set (default)
//!
void MEModuleTree::highlightHistoryModules()
{

    // read infos from mapeditor.xml file and store it in a map
    for (unsigned int k = 0; k < m_mainHandler->moduleHistory.count(); k++)
    {
        QString modulename = m_mainHandler->moduleHistory[k].section(":", 1, 1).section("(", 0, 0);
        QString categoryname = m_mainHandler->moduleHistory[k].section("(", 1, 1).section(")", 0, 0);

        m_usedModules << modulename + " (" + categoryname + ")";
    }
}

//!
//! Find a tree item for a given category item & modulename
//!
QTreeWidgetItem *MEModuleTree::findModule(QTreeWidgetItem *categoryItem, const QString &name)
{

    for (int i = 0; i < categoryItem->childCount(); i++)
    {
        QTreeWidgetItem *item = categoryItem->child(i);
        if (item->text(0) == name || item->text(0).section(" ", 0, 0) == name)
        {
            return item;
        }
    }

    return NULL;
}

//!
//! Find a category item for a given root item & category name
//!
QTreeWidgetItem *MEModuleTree::findCategory(QTreeWidgetItem *hostItem, const QString &name)
{

    for (int i = 0; i < hostItem->childCount(); i++)
    {
        QTreeWidgetItem *item = hostItem->child(i);
        if (item->text(0) == name)
            return item;
    }

    return NULL;
}

//!
//! Open/close categories or add modules
//!
void MEModuleTree::doubleClicked(QTreeWidgetItem *item, int)
{
    if (item == NULL || !m_mainHandler->isMaster())
        return;

    int deep = getDepth(item);

    // close all hosts or categories
    // show the selected one

    if (deep == 0)
    {
        for (int i = 0; i < topLevelItemCount(); i++)
            topLevelItem(i)->setExpanded(false);
    }

    else if (deep == 1)
    {
        QTreeWidgetItem *root = item->parent();
        for (int i = 0; i < root->childCount(); i++)
            root->child(i)->setExpanded(false);
    }

    else if (deep >= 2)
    {
        QString hostname, username, category, modulename;
        getHostUserCategoryName(item, &hostname, &username, &category, &modulename);

        // send message to controller to start the module
        QPointF pp = MEGraphicsView::instance()->getFreePos();

        m_mainHandler->requestNode(modulename, hostname, (int)pp.x(), (int)pp.y(), NULL, MEMainHandler::NORMAL);
    }
}

void MEModuleTree::executeVisibleModule()
{
    QTreeWidgetItem *exec = NULL;
    int count = 0;
    for (int i = 0; i < topLevelItemCount(); ++i)
    {
        QTreeWidgetItem *item = topLevelItem(i);
        if (item->isHidden())
            continue;

        while (item)
        {
            if (item->childCount() == 0)
            {
                exec = item;
                ++count;
                break;
            }

            int visibleChildren = 0;
            QTreeWidgetItem *child = NULL;
            for (int i = 0; i < item->childCount(); ++i)
            {
                if (!item->child(i)->isHidden())
                {
                    ++visibleChildren;
                    child = item->child(i);
                }
            }
            item = visibleChildren == 1 ? child : NULL;
        }
    }

    if (!exec || count > 1)
        return;

    MEUserInterface::instance()->resetModuleFilter();

    QString hostname, username, category, modulename;
    getHostUserCategoryName(exec, &hostname, &username, &category, &modulename);

    // send message to controller to start the module
    QPointF pp = MEGraphicsView::instance()->getFreePos();

    m_mainHandler->requestNode(modulename, hostname, (int)pp.x(), (int)pp.y(), NULL, MEMainHandler::NORMAL);
}

//!
//! Show all other modules if more... was clicked
//!
void MEModuleTree::clicked(QTreeWidgetItem *item, int)
{
    if (item == NULL)
        return;

    // below category, should be more...
    if (getDepth(item) == 2 && item->text(0) == "More...")
    {
        // loop over module items in category
        QTreeWidgetItem *p = item->parent();

        for (int i = 0; i < p->childCount(); i++)
        {
            QTreeWidgetItem *child = p->child(i);
            child->setHidden(false);
        }

        // hide more...
        item->setHidden(true);
    }

    else
    {
        int deep = getDepth(item);

        // show used nodes for a selected category
        if (deep == 1)
            emit showUsedCategory(item->text(0));

        // show used nodes for a selected module
        else if (deep > 1)
            emit showUsedNodes(item->parent()->text(0), item->text(0));
    }
}

void MEModuleTree::collapsed(QTreeWidgetItem *item)
{
    if (item != NULL && getDepth(item) > 0)
        item->setIcon(0, m_mainHandler->pm_folderclosed);
}

void MEModuleTree::expanded(QTreeWidgetItem *item)
{
    hideUnusedItems(item);
    if (item != NULL && getDepth(item) > 0)
        item->setIcon(0, m_mainHandler->pm_folderopen);
}

//!
//! Calculate the depth of an item in a tree hierarchy
//!
int MEModuleTree::getDepth(const QTreeWidgetItem *item) const
{
    int index = 0;

    const QTreeWidgetItem *it = item;
    while (it->parent())
    {
        it = it->parent();
        index++;
    }

    return index;
}

//!
//! User has moved a module into the canvasArea, update module list
//!
void MEModuleTree::moduleUseNotification(const QString &modname)
{
    const QString all = "All";
    bool hide = m_mainHandler->cfg_HideUnusedModules;
    QString categoryName = modname.section(":", 0, 0);
    QString moduleName = modname.section(":", 1, 1);
    m_usedModules += QString("%1 (%2)").arg(moduleName, categoryName);

    // get info
    for (int i = 0; i < topLevelItemCount(); i++)
    {
        QTreeWidgetItem *root = topLevelItem(i);

        // add module to category ALL
        QTreeWidgetItem *categoryItem = findCategory(root, all);
        if (categoryItem)
        {
            QTreeWidgetItem *moduleItem = findModule(categoryItem, moduleName);
            if (moduleItem && hide)
                moduleItem->setFont(0, MEMainHandler::s_boldFont);
        }

        // change module to beginner mode
        categoryItem = findCategory(root, categoryName);
        if (categoryItem)
        {
            QTreeWidgetItem *moduleItem = findModule(categoryItem, moduleName);
            if (moduleItem && hide)
                moduleItem->setFont(0, MEMainHandler::s_boldFont);
        }
    }
}

//!
//! Change browser items after the expert mode has changed
//!
void MEModuleTree::changeBrowserItems()
{
    // loop over all items ins the browser tree
    // but only for the local host

    for (int i = 0; i < MEHostListHandler::instance()->getNoOfHosts(); ++i)
    {
        MEHost *host = MEHostListHandler::instance()->getHostAt(i);
        QTreeWidgetItem *root = host->getModuleRoot();

        // loop over all categories
        for (int j = 0; j < root->childCount(); ++j)
        {
            QTreeWidgetItem *categoryItem = root->child(j);
            hideUnusedItems(categoryItem);
        }
    }
}

//!
//! Find all matching categories and modules for a given string, fill the filterTree
//!
void MEModuleTree::showMatchingItems(const QString &text)
{
    // clear all items
    clear();

    for (int i = 0; i < MEHostListHandler::instance()->getNoOfHosts(); ++i)
    {
        // make root item with host color
        MEHost *host = MEHostListHandler::instance()->getHostAt(i);

        QTreeWidgetItem *root = new MEModuleTreeItem(MEUserInterface::instance()->getFilterTree());
        root->setText(0, host->getText());
        root->setIcon(0, host->getIcon());
        QBrush brush = root->foreground(0);

        int hue, s, v;
        host->getColor().getHsv(&hue, &s, &v);
        QColor color_dark;
        color_dark.setHsv(hue, 255, v);
        brush.setColor(color_dark);
        root->setForeground(0, brush);
        root->setExpanded(true);

        // insert modules and categories that match
        // if a category matches show all items

        bool showHost = false;
        foreach (MECategory *category, host->catList)
        {
            bool match = false;
            QString name = category->getName();
            if (name != "All")
            {
                // create category item
                QTreeWidgetItem *categoryItem = new QTreeWidgetItem(root);
                categoryItem->setText(0, name);
                categoryItem->setToolTip(0, s_moduleHelp.value(name));

                // does the category match ?
                if (name.contains(text, Qt::CaseInsensitive))
                {
                    categoryItem->setHidden(false);
                    categoryItem->setExpanded(true);

                    // show all modules
                    int nc = category->moduleList.count();
                    for (int i = 0; i < nc; i++)
                    {
                        QString mname = category->moduleList.at(i);
                        QString tooltip = s_moduleHelp.value(name + "/" + mname);
                        QTreeWidgetItem *moduleItem = new MEModuleTreeItem(categoryItem);
                        moduleItem->setText(0, mname);
                        moduleItem->setIcon(0, m_mainHandler->pm_bullet);
                        moduleItem->setToolTip(0, tooltip);
                    }
                    categoryItem->sortChildren(0, Qt::AscendingOrder);
                }

                else
                {
                    categoryItem->setHidden(true);

                    // insert matching module names
                    int nc = category->moduleList.count();
                    for (int i = 0; i < nc; i++)
                    {
                        QString mname = category->moduleList.at(i);
                        QString tooltip = s_moduleHelp.value(name + "/" + mname);
                        if (mname.contains(text, Qt::CaseInsensitive) || tooltip.contains(text, Qt::CaseInsensitive))
                        {
                            match = true;
                            QTreeWidgetItem *moduleItem = new MEModuleTreeItem(categoryItem);
                            moduleItem->setText(0, mname);
                            moduleItem->setIcon(0, m_mainHandler->pm_bullet);
                            moduleItem->setToolTip(0, tooltip);
                        }
                    }

                    // make selection visible
                    if (match)
                    {
                        categoryItem->setHidden(false);
                        categoryItem->setExpanded(true);
                        categoryItem->sortChildren(0, Qt::AscendingOrder);
                    }
                }
            }
            if (match)
                showHost = true;
        }

        if (showHost)
        {
            root->setHidden(false);
            root->setExpanded(true);
            root->sortChildren(0, Qt::AscendingOrder);
        }
    }
    MEUserInterface::instance()->switchModuleTree(this, true);
    setHeaderLabel("Filtered Module List (" + text + ")");
}

//!
//! Show the full module list again
//!
void MEModuleTree::restoreList()
{
    setHeaderLabel("Module List");
    MEUserInterface::instance()->switchModuleTree(this, false);
}

//!
//! Analyze right mouse button press event
//!
void MEModuleTree::contextMenuEvent(QContextMenuEvent *e)
{
    // get clicked item
    m_clickedItem = itemAt(e->x(), e->y());

    int deep = getDepth(m_clickedItem);

    // show host menus
    // show menu only if host is not localhost
    if (deep == 0)
    {
        QStringList list = m_clickedItem->text(0).split("@");
        if (list[1] != m_mainHandler->localHost)
            m_hostMenu->popup(e->globalPos());
    }

    // show category description
    else if (deep == 1)
    {
        m_clickedCategory = m_clickedItem;
        m_currentModuleName = "";
        m_categoryMenu->popup(e->globalPos());
    }

    // show category/module description
    else
    {
        m_clickedCategory = m_clickedItem->parent();
        m_currentModuleName = m_clickedItem->text(0);
        m_moduleMenu->popup(e->globalPos());
    }
}

//!
//! Only CopyAction allowed ! (seems to be a problem on MacOS only)
//!
void MEModuleTree::startDrag(Qt::DropActions /*supportedActions*/)
{
    if (m_mainHandler->isMaster())
        QAbstractItemView::startDrag(Qt::CopyAction);
}

void MEModuleTree::dragEnterEvent(QDragEnterEvent *event)
{
    // don't allow drops from outside the application
    if (event->source() == NULL)
        return;

    if (event->mimeData()->hasText())
        event->accept();
}

void MEModuleTree::dragLeaveEvent(QDragLeaveEvent *event)
{
    event->accept();
}

void MEModuleTree::dragMoveEvent(QDragMoveEvent *event)
{
    // don't allow drops from outside the application
    if (event->source() == NULL)
        return;

    if (event->source() == this)
        event->ignore();

    else if (event->mimeData()->hasText())
        event->accept();
}

//!
//! Remove favorite from list when dropped in module list
//!
bool MEModuleTree::dropMimeData(QTreeWidgetItem *parent, int index, const QMimeData *data, Qt::DropAction action)
{
    Q_UNUSED(index);
    Q_UNUSED(parent);

    if (action == Qt::IgnoreAction)
        return true;

    if (data->hasText())
    {
        QString text = data->text();
        QString combiname = text.section(':', 3, 3) + ":" + text.section(':', 2, 2);
        MEFavoriteListHandler::instance()->removeFavorite(combiname);
        return true;
    }

    return false;
}

bool MEModuleTree::getHostUserCategoryName(const QTreeWidgetItem *item,
                                           QString *h, QString *u, QString *c, QString *n) const
{
    if (getDepth(item) < 2)
        return false;

    // get parent items
    QTreeWidgetItem *categoryItem = item->parent();
    QTreeWidgetItem *root = categoryItem->parent();

    // get host and username
    QStringList list = root->text(0).split("@");
    QString username = "unknown";
    QString hostname = MEHostListHandler::instance()->getIPAddress(list[1]);

    // get the category name
    QString category = categoryItem->text(0);
    QString modulename = item->text(0);
    if (category == "All")
    {
        category = modulename.section("(", 1, 1).section(")", 0, 0);
        modulename = modulename.section(" ", 0, 0);
    }

    if (h)
        *h = hostname;
    if (u)
        *u = username;
    if (c)
        *c = category;
    if (n)
        *n = modulename;

    return true;
}

//!
//! Create the mime data for dragging
//!
QMimeData *MEModuleTree::mimeData(const QList<QTreeWidgetItem *> dragList) const
{

    QByteArray encodedData;
    QDataStream stream(&encodedData, QIODevice::WriteOnly);

    foreach (QTreeWidgetItem *item, dragList)
    {
        int depth = 0;
        QTreeWidgetItem *it = item;
        while (it->parent())
        {
            it = it->parent();
            depth++;
        }

        // only modules can be dragged
        if (depth >= 2)
        {
            // create string & drag object
            QString h, u, c, n;
            getHostUserCategoryName(item, &h, &u, &c, &n);
            stream << h + ":" + u + ":" + c + ":" + n;

            // create a mime source
            QMimeData *mimeData = new QMimeData;
            mimeData->setData("application/x-qabstractitemmodeldatalist", encodedData);
            return mimeData;
        }
    }
    return NULL;
}

//!
//! Show information for module in COVISE helpviewer
//!
void MEModuleTree::infoCB()
{
    m_mainHandler->showModuleHelp(m_clickedCategory->text(0), m_currentModuleName);
}

//!
//! Start module in debugger
//!
void MEModuleTree::execDebugCB()
{
    QString hostname, username, category, modulename;
    getHostUserCategoryName(m_clickedItem, &hostname, &username, &category, &modulename);

    // send message to controller to start the module
    QPointF pp = MEGraphicsView::instance()->getFreePos();

    m_mainHandler->requestNode(modulename, hostname, (int)pp.x(), (int)pp.y(), NULL, MEMainHandler::DEBUG);
}

//!
//! Start module with memory checker
//!
void MEModuleTree::execMemcheckCB()
{
    QString hostname, username, category, modulename;
    getHostUserCategoryName(m_clickedItem, &hostname, &username, &category, &modulename);

    // send message to controller to start the module
    QPointF pp = MEGraphicsView::instance()->getFreePos();

    m_mainHandler->requestNode(modulename, hostname, (int)pp.x(), (int)pp.y(), NULL, MEMainHandler::MEMCHECK);
}

//!
//! Delete a host from tree
//!
void MEModuleTree::removeHostCB()
{
    /*if(m_mainHandler->m_mirrorMode >=2)
   {
      m_mainHandler->printMessage("You can't delete hosts when you are mirroring pipelines.\n Stop mirroring before");
      return;
   }*/

    // get pressed host item (hostname, username)
    QStringList list = m_clickedItem->text(0).split("@");

    // build message
    QStringList text;
    QString hostname = MEHostListHandler::instance()->getIPAddress(list[1]);
    text << "RMV_HOST" << hostname << list[0] << "NONE";
    QString tmp = text.join("\n");

    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, tmp);
}

OperationWaiter::OperationWaiter()
    : m_done(false)
{
}

bool OperationWaiter::wait(int limit)
{
    m_done = false;

    QTime t;
    t.start();
    while (!m_done && (limit < 0 || t.elapsed() < limit))
    {
        usleep(10000);
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }

    if (m_done)
        qDebug() << "operation finished successfully in" << t.elapsed() << "ms";
    else
        qDebug() << "operation given up after" << t.elapsed() << "ms";

    return m_done;
}

void OperationWaiter::finished()
{
    m_done = true;
}
