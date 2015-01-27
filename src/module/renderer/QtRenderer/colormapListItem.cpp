/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/*
    colormapListItem.cpp


*/

#include "colormapListItem.h"

// Data //
//
#include "InvObjectList.h"

// Tree //
//
#include "colormapListWidget.h"
#include "InvMain.h"

//################//
// CONSTRUCTOR    //
//################//

colormapListItem::colormapListItem(QTreeWidgetItem *parent, QPixmap *dataElement, QTreeWidgetItem *fosterParent)
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

colormapListItem::~colormapListItem()
{
}

//################//
// FUNCTIONS      //
//################//

void
colormapListItem::init()
{

    // Default settings //
    //
    setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    // Text //
    //
    /*if(dataElement_)
	setText(0, dataElement_->getName());
	else
	setText(0, tr("ObjectsRoot"));*/

    /*TODO if(dataElement_->isElementSelected())
	{
		setSelected(true);
	}*/
}

void
colormapListItem::setData(int column, int role, const QVariant &value)
{
    //	qDebug() << "role: " << role;

    // View: selected ->  Data: select //
    //
    if (role == Qt::UserRole /*+colormapListWidget::PTR_Selection*/)
    {
        if (column != 0) // only first column
        {
            return;
        }

        //renderer->viewer->objectListCB(dataElement_,value.toBool());
    }

    QTreeWidgetItem::setData(column, role, value);
}

colormapListItem *colormapListItem::findChild(const QString &name)
{
    //	if(name==dataElement_->getName())
    {
        return this;
    }
    colormapListItem *item;
    for (int i = 0; i < childCount(); i++)
    {
        item = dynamic_cast<colormapListItem *>(QTreeWidgetItem::child(i));
        if (item)
        {
            colormapListItem *it = item->findChild(name);
            if (it)
                return it;
        }
    }
    return NULL;
}
