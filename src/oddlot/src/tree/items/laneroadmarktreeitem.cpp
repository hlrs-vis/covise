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

#include "laneroadmarktreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"

// Tree //
//
#include "lanetreeitem.hpp"

LaneRoadMarkTreeItem::LaneRoadMarkTreeItem(LaneTreeItem *parent, LaneRoadMark *laneRoadMark, QTreeWidgetItem *fosterParent)
    : ProjectTreeItem(parent, laneRoadMark, fosterParent)
    , laneTreeItem_(parent)
    , laneRoadMark_(laneRoadMark)
{
    init();
}

LaneRoadMarkTreeItem::~LaneRoadMarkTreeItem()
{
}

void
LaneRoadMarkTreeItem::init()
{
    updateName();
}

void
LaneRoadMarkTreeItem::updateName()
{
    QString text(tr(""));
    text.append(QString("%1: ").arg(laneRoadMark_->getSOffset()));
    text.append(LaneRoadMark::parseRoadMarkTypeBack(laneRoadMark_->getRoadMarkType()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
LaneRoadMarkTreeItem::updateObserver()
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
    int changes = laneRoadMark_->getRoadMarkChanges();
    if ((changes & LaneRoadMark::CLR_OffsetChanged)
        || (changes & LaneRoadMark::CLR_TypeChanged))
    {
        // Change of the road coordinate s //
        //
        updateName();
    }
}
