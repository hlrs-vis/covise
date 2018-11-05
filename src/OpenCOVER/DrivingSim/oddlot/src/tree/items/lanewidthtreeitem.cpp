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

#include "lanewidthtreeitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/laneborder.hpp"

// Tree //
//
#include "lanetreeitem.hpp"


LaneWidthTreeItem::LaneWidthTreeItem(LaneTreeItem *parent, LaneWidth *laneWidth, QTreeWidgetItem *fosterParent)
	: ProjectTreeItem(parent, laneWidth, fosterParent)
	, laneTreeItem_(parent)
	, laneWidth_(laneWidth)
{
	init();
}


LaneWidthTreeItem::~LaneWidthTreeItem()
{
}

void
LaneWidthTreeItem::init()
{
    updateName();
}

void
LaneWidthTreeItem::updateName()
{
    QString text(tr(""));
    text.append(QString("%1: ").arg(laneWidth_->getSOffset()));
    text.append(QString("n = %1").arg(laneWidth_->getDegree()));
    setText(0, text);
}

//##################//
// Observer Pattern //
//##################//

void
LaneWidthTreeItem::updateObserver()
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
    int changes = laneWidth_->getLaneWidthChanges();
    if ((changes & LaneWidth::CLW_OffsetChanged)
        || (changes & LaneWidth::CLW_WidthChanged))
    {
        // Change of the road coordinate s //
        //
        updateName();
    }
}
