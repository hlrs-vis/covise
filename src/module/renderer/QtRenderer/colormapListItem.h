/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COLORMAPLISTITEM_H
#define COLORMAPLISTITEM_H

#include <QObject>
#include <QTreeWidgetItem>
#include <QString>
#include <InvObjectList.h>

class colormapListWidget;

class colormapListItem : public QObject, public QTreeWidgetItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit colormapListItem(QTreeWidgetItem *parent, QPixmap *p, QTreeWidgetItem *fosterParent = NULL);
    virtual ~colormapListItem();

    virtual void setData(int column, int role, const QVariant &value);

    // Tree //
    //

    // DataElement //
    //
    QPixmap *getDataElement() const
    {
        return dataElement_;
    }

    // ParentGraphElement //
    //
    QTreeWidgetItem *getParentTreeItem() const
    {
        return parentTreeItem_;
    }

    colormapListItem *findChild(const QString &name);

protected:
private:
    colormapListItem(); /* not allowed */
    colormapListItem(const colormapListItem &); /* not allowed */
    colormapListItem &operator=(const colormapListItem &); /* not allowed */

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
    QPixmap *dataElement_;

    //################//
    // STATIC         //
    //################//
};

#endif
