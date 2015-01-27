/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/*
    objTreeItem.cpp


*/

#include "objTreeItem.h"

// Data //
//
#include "InvObjectList.h"

// Tree //
//
#include "objTreeWidget.h"
#include "InvMain.h"

//################//
// CONSTRUCTOR    //
//################//

objTreeItem::objTreeItem(QTreeWidgetItem *parent, InvObject *dataElement, QTreeWidgetItem *fosterParent)
    : QObject()
    , QTreeWidgetItem()
    , parentTreeItem_(parent)
    , dataElement_(dataElement)
{
    // Append to fosterParent if given, parent otherwise //
    //
    if (fosterParent)
    {
        fosterParent->addChild(this);
    }
    else
    {
        parent->addChild(this);
    }

    // Init //
    //
    init();
}

objTreeItem::~objTreeItem()
{
}

//################//
// FUNCTIONS      //
//################//

void
objTreeItem::init()
{

    // Default settings //
    //
    setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    // Text //
    //
    if (dataElement_)
        setText(0, dataElement_->getName());
    else
        setText(0, tr("ObjectsRoot"));

    /*TODO if(dataElement_->isElementSelected())
	{
		setSelected(true);
	}*/
}

void
objTreeItem::setData(int column, int role, const QVariant &value)
{
    //	qDebug() << "role: " << role;

    // View: selected ->  Data: select //
    //
    if (role == Qt::UserRole /*+objTreeWidget::PTR_Selection*/)
    {
        if (column != 0) // only first column
        {
            return;
        }

        renderer->viewer->objectListCB(dataElement_, value.toBool());
    }

    QTreeWidgetItem::setData(column, role, value);
}

objTreeItem *objTreeItem::findChild(const QString &name)
{
    if (name == dataElement_->getName())
    {
        return this;
    }
    objTreeItem *item;
    for (int i = 0; i < childCount(); i++)
    {
        item = dynamic_cast<objTreeItem *>(QTreeWidgetItem::child(i));
        if (item)
        {
            objTreeItem *it = item->findChild(name);
            if (it)
                return it;
        }
    }
    return NULL;
}
