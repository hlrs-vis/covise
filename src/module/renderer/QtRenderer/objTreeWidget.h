/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TREEWIDGET_H
#define TREEWIDGET_H

#include <QTreeWidget>
#include <QModelIndex>

class objTreeItem;
class InvMain;

class objTreeWidget : public QTreeWidget
{
    Q_OBJECT

public:
    objTreeWidget(InvMain *im, QWidget *parent = 0);
    ~objTreeWidget();

protected:
private:
    objTreeWidget(); /* not allowed */
    objTreeWidget(const objTreeWidget &); /* not allowed */
    objTreeWidget &operator=(const objTreeWidget &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    void selectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

    QTreeWidgetItem *rootItem;

    objTreeItem *findItem(const QString &name) const;

private:
    InvMain *renderer;
};
//! [0]

#endif
