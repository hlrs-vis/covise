/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/13/2010
**
**************************************************************************/

#include "objecttreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/objectobject.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

ObjectTreeItem::ObjectTreeItem(RoadTreeItem *parent, Object *object, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, object, fosterParent)
    , object_(object)
{
    init();
}

ObjectTreeItem::~ObjectTreeItem()
{
}

void
ObjectTreeItem::init()
{
    updateName();
}

void
ObjectTreeItem::updateName()
{
    QString text = object_->getName();
    text.append(" (");
    text.append(object_->getType());
    text.append(QString(") %1").arg(object_->getSStart()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
ObjectTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Object //
    //
    int changes = object_->getObjectChanges();

    if (changes & Object::CEL_ParameterChange)
    {
        updateName();
    }
}
