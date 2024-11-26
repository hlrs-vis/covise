/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QHeaderView>
#include <QDebug>

#include "MEDataTree.h"
#include "MEDataViewer.h"
#include "MEDataObject.h"
#include "MEDataArray.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"
#include "widgets/MEUserInterface.h"

#include <covise/covise_msg.h>

/*!
    \class MEDataTreeItem
    \brief Subclassed QTreeWidgetItem used by MEDataTree

*/

// used for top item showing the host
MEDataTreeItem::MEDataTreeItem(MEDataTree *item, const QString &text, const QColor &color)
    : QTreeWidgetItem(item)
    , m_dataObject(NULL)
    , m_dataArray(NULL)
    , m_dataObjectType(0)
    , m_color(color)
{
    setText(0, text);
    setIcon(0, MEMainHandler::instance()->pm_host);
    setFlags(Qt::ItemIsEnabled);
    m_port = NULL;
}

// used for sub data objects
MEDataTreeItem::MEDataTreeItem(MEDataTreeItem *item, const QString &text)
    : QTreeWidgetItem(item)
    , m_dataObject(NULL)
    , m_dataArray(NULL)
    , m_dataObjectType(0)
{
    setText(0, text);
    m_color = item->getColor();
    m_port = item->getPort();
    setFlags(Qt::ItemIsEnabled);
}

// root item for a covise port (data object name)
MEDataTreeItem::MEDataTreeItem(MEDataTreeItem *item, const QString &text, MEDataPort *port)
    : QTreeWidgetItem(item)
    , m_dataObject(NULL)
    , m_dataArray(NULL)
    , m_dataObjectType(0)
{
    setText(0, text);
    setFlags(Qt::ItemIsEnabled);
    m_color = item->getColor();
    m_port = port;
    item->setIcon(0, MEMainHandler::instance()->pm_folderclosed);
    setIcon(0, MEMainHandler::instance()->pm_folderclosed);
}

//!
//! delete a data tree item and the corresponding data object / data array
//!
MEDataTreeItem::~MEDataTreeItem()
{
    if (m_dataObject)
        delete m_dataObject;

    if (m_dataArray)
        delete m_dataArray;
}

//!
//! pipeline was executed, update item name and content
//!
void MEDataTreeItem::updateItem()
{
// set the new data name
    QString tmp = text(0);
    if (tmp.contains(":: "))
    {
        tmp = tmp.section(':', -1);
        tmp = tmp.remove(0, 1);
    }

    // update content of windows data object
    if (m_dataObject)
        m_dataObject->update(tmp);

    // update content of windows data array
    if (m_dataArray)
        m_dataArray->update();
}

//!
//! create/show the data object info or data array
//!
void MEDataTreeItem::showItemContent()
{

    if (m_dataArray)
        MEDataViewer::instance()->showArray(m_dataArray);

    else if (m_dataObject)
        MEDataViewer::instance()->showObject(m_dataObject);

    else if (m_dataObjectType == MEDataObject::POINTER)
        m_dataObject = MEDataViewer::instance()->createObject(this);

    else if (m_dataObjectType == MEDataObject::ARRAY)
        m_dataArray = MEDataViewer::instance()->createArray(this);

    MEDataTree::instance()->clearSelection();
    setSelected(true);
}

/*!
    \class MEDataTree
    \brief Tree widget showing all available data object

    This class is part of the MEDataViewer
*/

MEDataTree::MEDataTree(QWidget *parent)
    : QTreeWidget(parent)
{
}

//!
//1 init the tree widget
//!
void MEDataTree::init()
{

    const char *text2 = "<h4>Data Object Browser</h4>"
                        "<p>Data objects are created when a map is executed. </p>"
                        "<p>The names of the data objects are generated generically after the map was executed once. "
                        "After a new execution the list is updated with the new names. </p>"
                        "<p>Click on a data object in the browser. More information is shown in the <b>DataViewer</b> window "
                        "on the right side of the <b>Map Editor</b></p>";

    setRootIsDecorated(true);
    setSelectionMode(QAbstractItemView::SingleSelection);
    setSelectionBehavior(QAbstractItemView::SelectItems);
    sortByColumn(0, Qt::AscendingOrder);
    setSortingEnabled(true);

    setColumnCount(1);
    setHeaderLabel("Data Object List");

    connect(this, SIGNAL(itemDoubleClicked(QTreeWidgetItem *, int)), this, SLOT(doubleClicked(QTreeWidgetItem *, int)));
    connect(this, SIGNAL(itemClicked(QTreeWidgetItem *, int)), this, SLOT(activated(QTreeWidgetItem *, int)));
    connect(this, SIGNAL(itemCollapsed(QTreeWidgetItem *)), this, SLOT(collapsed(QTreeWidgetItem *)));
    connect(this, SIGNAL(itemExpanded(QTreeWidgetItem *)), this, SLOT(expanded(QTreeWidgetItem *)));

    setWhatsThis(text2);
}

MEDataTree *MEDataTree::instance()
{
    static MEDataTree *singleton = 0;
    if (singleton == 0)
    {
        singleton = new MEDataTree();
        singleton->init();
    }

    return singleton;
}

MEDataTree::~MEDataTree()
{
}

//!
//! change pixmap after an item was collapsed
//!
void MEDataTree::collapsed(QTreeWidgetItem *item)
{
    if (item != NULL && item != topLevelItem(0))
        item->setIcon(0, MEMainHandler::instance()->pm_folderclosed);
}

//!
//! change pixmap after an item was expanded
//!
void MEDataTree::expanded(QTreeWidgetItem *item)
{
    if (item)
    {
        if (item->isDisabled())
            item->setExpanded(false);

        else
        {
            item->setIcon(0, MEMainHandler::instance()->pm_folderopen);
            if (getDepth(item) >= 2)
            {
                QTreeWidgetItem *dummy = item->child(0);
                if (dummy && dummy->text(0) == "dummy")
                    item->removeChild(dummy);
                MEDataTreeItem *it = static_cast<MEDataTreeItem *>(item);
                it->showItemContent();
            }
        }
    }
}

//!
//! item was clicked
//!
void MEDataTree::activated(QTreeWidgetItem *item, int)
{
    if (item && getDepth(item) >= 2 && !item->isDisabled())
    {
        MEDataTreeItem *it = static_cast<MEDataTreeItem *>(item);
        it->showItemContent();
    }
}

//!
//! Open/close items
//!
void MEDataTree::doubleClicked(QTreeWidgetItem *item, int)
{
    if (item == NULL || item->isDisabled())
        return;

    int deep = getDepth(item);

    // close all hosts or categories
    // show the selected one

    if (deep == 0)
    {
        for (int i = 0; i < topLevelItemCount(); i++)
        {
            QTreeWidgetItem *root = topLevelItem(i);
            if (root == item)
                item->setExpanded(true);
            else
                root->setExpanded(false);
        }
    }

    else
    {
        if (item->childCount() == 0)
            return;

        QTreeWidgetItem *root = item->parent();
        for (int i = 0; i < root->childCount(); i++)
        {
            QTreeWidgetItem *categoryItem = root->child(i);
            if (categoryItem == item)
                item->setExpanded(true);
            else
                categoryItem->setExpanded(false);
        }
    }
}

//!
//! calculate the depth of an item in a tree hierarchy
//!
int MEDataTree::getDepth(QTreeWidgetItem *item)
{
    int depth = 0;

    QTreeWidgetItem *it = item;
    while (it->parent())
    {
        it = it->parent();
        depth++;
    }

    return depth;
}
