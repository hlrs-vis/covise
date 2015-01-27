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
#ifdef YAC
#include "yac/coQTSendBuffer.h"
#endif

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
    , m_index(-1)
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
    , m_index(-1)
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
    , m_index(-1)
{
    setText(0, text);
    setFlags(Qt::ItemIsEnabled);
    m_color = item->getColor();
    m_port = port;
    item->setIcon(0, MEMainHandler::instance()->pm_folderclosed);
    setIcon(0, MEMainHandler::instance()->pm_folderclosed);
}

// root item for a yac port
MEDataTreeItem::MEDataTreeItem(MEDataTreeItem *item, const QString &text, int id)
    : QTreeWidgetItem(item)
    , m_dataObject(NULL)
    , m_dataArray(NULL)
    , m_dataObjectType(0)
{
    m_index = id; // index inside data object, only used under YAC
    setText(0, text);
    setFlags(Qt::ItemIsEnabled);
    m_color = item->getColor();
    m_port = item->getPort();
    setIcon(0, MEMainHandler::instance()->pm_folderclosed);
    item->setIcon(0, MEMainHandler::instance()->pm_folderclosed);
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
#ifdef YAC

    QString tmp = parent()->text(0) + " (" + text(1) + "," + text(2) + "," + text(3) + ")";

#else

    QString tmp = text(0);
    if (tmp.contains(":: "))
    {
        tmp = tmp.section(':', -1);
        tmp = tmp.remove(0, 1);
    }
#endif

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

#ifdef YAC

    // request data object information
    covise::coSendBuffer sb;
    if (m_index == -1)
    {
        int i1 = text(1).toInt();
        int i2 = text(2).toInt();
        int i3 = text(3).toInt();
        sb << i1 << i2 << i3;
        MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_GET_OBJ_INFO, sb);
    }

    else
    {
        int i1 = parent()->text(1).toInt();
        int i2 = parent()->text(2).toInt();
        int i3 = parent()->text(3).toInt();
        sb << i1 << i2 << i3 << m_index << 0 << 199;
        MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_GET_OBJ_ARRAY, sb);
    }

#else

    else if (m_dataObjectType == MEDataObject::POINTER)
        m_dataObject = MEDataViewer::instance()->createObject(this);

    else if (m_dataObjectType == MEDataObject::ARRAY)
        m_dataArray = MEDataViewer::instance()->createArray(this);
#endif

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

#ifdef YAC

    setColumnCount(6);
    QStringList headers;
    headers << "Data Objects"
            << "PortID"
            << "ModID"
            << "SeqNo"
            << "Block"
            << "Timestep";
    setHeaderLabels(headers);
    header()->setResizeMode(QHeaderView::ResizeToContents);

    // add the always existing item for lost&found
    lostfound = new MEDataTreeItem(this, "Lost&Found", QColor(100, 200, 100));

#else

    setColumnCount(1);
    setHeaderLabel("Data Object List");
#endif

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

//!
//! get a tree item for a given object ID (used by YAC)
//!
MEDataTreeItem *MEDataTree::findObject(int i1, int i2, int i3)
{
    // store object ids for search process
    m_id1 = i1;
    m_id2 = i2;
    m_id3 = i3;

    // find item (recursive loop)
    MEDataTreeItem *it = NULL;

    for (int i = 0; i < topLevelItemCount(); i++)
    {
        QTreeWidgetItem *top = topLevelItem(i);
        it = search(top);
        if (it != NULL)
            return it;
    }

    if (it == NULL)
    {
        QString text = "UI_OBJ_ARRAY : Can't find requested array for object " + QString::number(m_id1) + " " + QString::number(m_id2) + " " + QString::number(m_id3);
        MEUserInterface::instance()->printMessage(text);
    }
    return NULL;
}

//!
//! get an array item for a given object ID and index
//!
MEDataTreeItem *MEDataTree::findArray(MEDataTreeItem *item, int index)
{

    // find tree item
    // start point is this tree item
    MEDataTreeItem *child = NULL;

    for (int i = 0; i < item->childCount(); i++)
    {
        child = static_cast<MEDataTreeItem *>(item->child(i));

        if (child->getObjType() == MEDataObject::ARRAY && child->getIndex() == index)
            return child;
    }

    if (child == NULL)
    {
        QString text = "UI_OBJ_ARRAY : Can't find requested array for object " + item->text(1) + " " + item->text(2) + " " + item->text(3);
        MEUserInterface::instance()->printMessage(text);
    }
    return NULL;
}

//!
//! search tree items for a certain object id (used by YAC)
//!
MEDataTreeItem *MEDataTree::search(QTreeWidgetItem *item)
{

    QTreeWidgetItem *child = NULL;
    for (int i = 0; i < item->childCount(); i++)
    {
        child = item->child(i);
        int i1 = child->text(1).toInt();
        int i2 = child->text(2).toInt();
        int i3 = child->text(3).toInt();

        if (i1 == m_id1 && i2 == m_id2 && i3 == m_id3)
            return static_cast<MEDataTreeItem *>(child);

        else
        {
            MEDataTreeItem *it = search(child);
            if (it != NULL)
                return it;
        }
    }

    return NULL;
}
