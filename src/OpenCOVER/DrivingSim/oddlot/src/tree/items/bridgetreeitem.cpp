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

#include "bridgetreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/bridgeobject.hpp"

// Tree //
//
#include "roadtreeitem.hpp"

BridgeTreeItem::BridgeTreeItem(RoadTreeItem *parent, Bridge *bridge, QTreeWidgetItem *fosterParent)
    : SectionTreeItem(parent, bridge, fosterParent)
    , bridge_(bridge)
{
    init();
}

BridgeTreeItem::~BridgeTreeItem()
{
}

void
BridgeTreeItem::init()
{
    updateName();
}

void
BridgeTreeItem::updateName()
{
    QString text = bridge_->getName();
    text.append(" (");
    text.append(bridge_->getType());
    text.append(QString(") %1").arg(bridge_->getSStart()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
BridgeTreeItem::updateObserver()
{

    // Parent //
    //
    SectionTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Bridge //
    //
    int changes = bridge_->getBridgeChanges();

    if (changes & Bridge::CEL_ParameterChange)
    {
        updateName();
    }
}
