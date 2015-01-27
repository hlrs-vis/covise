/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TREEITEM_H
#define TREEITEM_H

#include <QObject>
#include <QTreeWidgetItem>
#include <QString>
#include <InvObjectList.h>

class objTreeWidget;

class objTreeItem : public QObject, public QTreeWidgetItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit objTreeItem(QTreeWidgetItem *parent, InvObject *dataElement, QTreeWidgetItem *fosterParent = NULL);
    virtual ~objTreeItem();

    virtual void setData(int column, int role, const QVariant &value);

    // Tree //
    //

    // DataElement //
    //
    InvObject *getDataElement() const
    {
        return dataElement_;
    }

    // ParentGraphElement //
    //
    QTreeWidgetItem *getParentTreeItem() const
    {
        return parentTreeItem_;
    }

    objTreeItem *findChild(const QString &name);

protected:
private:
    objTreeItem(); /* not allowed */
    objTreeItem(const objTreeItem &); /* not allowed */
    objTreeItem &operator=(const objTreeItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // ParentGraphElement //
    //
    QTreeWidgetItem *parentTreeItem_;

    // DataElement //
    //
    InvObject *dataElement_;

    //################//
    // STATIC         //
    //################//
};

#endif
