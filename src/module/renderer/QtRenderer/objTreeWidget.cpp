/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QtGui>

#include "objTreeItem.h"
#include "objTreeWidget.h"

//################//
// CONSTRUCTOR    //
//################//

objTreeWidget::objTreeWidget(InvMain *im, QWidget *parent)
    : QTreeWidget(parent)
{
    renderer = im;
    init();
}

objTreeWidget::~objTreeWidget()
{
    delete rootItem;
}

//################//
// FUNCTIONS      //
//################//

void
objTreeWidget::init()
{
    setSelectionMode(QAbstractItemView::ExtendedSelection);
    setUniformRowHeights(true);

    // Labels //
    //
    setHeaderLabels(QStringList() << tr("ObjectName"));

    // Root Items //
    //
    rootItem = invisibleRootItem();
}

//################//
// EVENTS         //
//################//

void
objTreeWidget::selectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{

    // Set the data of the item, so the item notices it's selection
    //
    foreach (QModelIndex index, deselected.indexes())
    {
        model()->setData(index, false, Qt::UserRole);
    }

    foreach (QModelIndex index, selected.indexes())
    {
        model()->setData(index, true, Qt::UserRole);
    }

    QTreeWidget::selectionChanged(selected, deselected);
}

objTreeItem *objTreeWidget::findItem(const QString &name) const
{
    objTreeItem *item;
    for (int i = 0; i < rootItem->childCount(); i++)
    {
        item = dynamic_cast<objTreeItem *>(rootItem->child(i));
        if (item)
        {
            objTreeItem *it = item->findChild(name);
            if (it)
                return it;
        }
    }
    return NULL;
}
