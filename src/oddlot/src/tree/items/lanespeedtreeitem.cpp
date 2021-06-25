/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/29/2010
**
**************************************************************************/

#include "lanespeedtreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanespeed.hpp"

// Tree //
//
#include "lanetreeitem.hpp"

LaneSpeedTreeItem::LaneSpeedTreeItem(LaneTreeItem *parent, LaneSpeed *laneSpeed, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, laneSpeed, fosterParent)
    , laneTreeItem_(parent)
    , laneSpeed_(laneSpeed)
{
    init();
}

LaneSpeedTreeItem::~LaneSpeedTreeItem()
{
}

void
LaneSpeedTreeItem::init()
{
    updateName();
}

void
LaneSpeedTreeItem::updateName()
{
    QString text(tr(""));
    text.append(QString("%1: ").arg(laneSpeed_->getSOffset()));
    text.append(QString("%1").arg(laneSpeed_->getMaxSpeed()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
LaneSpeedTreeItem::updateObserver()
{

    // Parent //
    //
    ProjectTreeItem::updateObserver();

    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Lane //
    //
    int changes = laneSpeed_->getLaneSpeedChanges();
    if ((changes & LaneSpeed::CLS_OffsetChanged)
        || (changes & LaneSpeed::CLS_MaxSpeedChanged))
    {
        // Change of the road coordinate s //
        //
        updateName();
    }
}
