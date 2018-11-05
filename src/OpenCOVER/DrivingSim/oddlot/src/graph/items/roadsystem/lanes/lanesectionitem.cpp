/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   10/15/2010
 **
 **************************************************************************/

#include "lanesectionitem.hpp"

 // Data //
 //
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/commands/lanesectioncommands.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/lanes/laneitem.hpp"
#include "src/graph/items/roadsystem/lanes/laneroaditem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/roadtextitem.hpp"
#include "src/graph/items/roadsystem/roaditem.hpp"

// Editor //
//
#include "src/graph/editors/laneeditor.hpp"

//################//
// CONSTRUCTOR    //
//################//

LaneSectionItem::LaneSectionItem(LaneEditor *laneEditor, RoadItem *parentRoadItem, LaneSection *laneSection)
	: SectionItem(parentRoadItem, laneSection)
	, laneEditor_(laneEditor)
	, laneSection_(laneSection)
{
	init();
}

LaneSectionItem::~LaneSectionItem()
{
}

void
LaneSectionItem::init()
{

	// Selection/Hovering //
	//
	setAcceptHoverEvents(true);
	setSelectable();

	// SectionItems //
	//
	foreach(Lane *lane, laneSection_->getLanes())
	{
		if (lane->getId() != 0)
		{
			laneItems_.insert(lane, new LaneItem(this, lane));
		}
	}
}

void
LaneSectionItem::createPath()
{
	QMap<Lane *, LaneItem *>::const_iterator it = laneItems_.constBegin();
	while (it != laneItems_.constEnd())
	{
		it.value()->createPath();
		it++;
	}
}

// LaneItems //
//
void 
LaneSectionItem::addLaneItem(LaneItem *item)
{
	laneItems_.insert(item->getLane(), item);
}

int 
LaneSectionItem::removeLaneItem(LaneItem *item)
{
	return laneItems_.remove(item->getLane());
}


LaneItem *
LaneSectionItem::getLaneItem(Lane *lane)
{
	return laneItems_.value(lane);
}

//##################//
// Handles          //
//##################//

/*! \brief .
*
*/
/*void
LaneSectionItem::rebuildMoveRotateHandles(bool delHandles)
{
	foreach(LaneItem *laneItem, laneItems_)
	{
		laneItem->rebuildMoveRotateHandles(delHandles);
	}
} */


/*! \brief .
*
*/
/*void
LaneSectionItem::deleteHandles()
{
	foreach(LaneItem *laneItem, laneItems_)
	{
		laneItem->deleteHandles();
	}
}*/

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
LaneSectionItem::updateObserver()
{
    // Parent //
    //
    SectionItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // LaneSection //
    //
    int changes = laneSection_->getLaneSectionChanges();

    if (changes & LaneSection::CLS_LanesChanged)
    {
        // A lane has been added.
        //
        foreach (Lane *lane, laneSection_->getLanes())
        {
            if ((lane->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (lane->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                if (lane->getId() != 0)
                {
                    laneItems_.insert(lane, new LaneItem(this, lane));
					dynamic_cast<LaneRoadItem *>(parentRoadItem_)->rebuildMoveRotateHandles(true);
                }
            }
        }
    }
	else if ((changes & LaneSection::CLS_LanesWidthsChanged) || (changes & LaneSection::CRS_LengthChange))
	{
		dynamic_cast<LaneRoadItem *>(parentRoadItem_)->rebuildMoveRotateHandles(true);
	}
}

//################//
// SLOTS          //
//################//

bool
LaneSectionItem::removeSection()
{
    RemoveLaneSectionCommand *command = new RemoveLaneSectionCommand(laneSection_, NULL);
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

void
LaneSectionItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = laneEditor_->getCurrentTool();
    if (tool == ODD::TLE_SELECT)
    {
        // parent: selection //
        SectionItem::mousePressEvent(event); // pass to baseclass
    }
    else
    {
        return; // prevent selection by doing nothing
    }
}

void
LaneSectionItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = laneEditor_->getCurrentTool();
    if (tool == ODD::TLE_SELECT)
    {
        // parent: selection //
        //SectionItem::mouseReleaseEvent(event); // do not pass selection to baseclass, we select individual lanes, not laneSections
    }
    else if (tool == ODD::TLE_ADD && (event->button() == Qt::LeftButton))
    {
        //if(laneSection_->getDegree() <= 1) // only lines can be split TODO find out, whether all lane widths are linear
        {
            // Split Section //
            //
            RSystemElementRoad *road = laneSection_->getParentRoad();
            double s = road->getSFromGlobalPoint(event->pos(), laneSection_->getSStart(), laneSection_->getSEnd());


			laneEditor_->getInsertSectionHandle()->hide();

            SplitLaneSectionCommand *command = new SplitLaneSectionCommand(laneSection_, s, NULL);
            getProjectGraph()->executeCommand(command);
        }
    }
    else if (tool == ODD::TLE_DEL && (event->button() == Qt::LeftButton))
    {
        removeSection();
    }
}

void
LaneSectionItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = laneEditor_->getCurrentTool();
    if (tool == ODD::TLE_SELECT)
    {
        // parent: selection //
        SectionItem::mouseMoveEvent(event); // pass to baseclass
    }
}

void
LaneSectionItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = laneEditor_->getCurrentTool();
    if (tool == ODD::TLE_SELECT)
    {
        setCursor(Qt::PointingHandCursor);
    }
    else if (tool == ODD::TLE_ADD)
    {
        setCursor(Qt::CrossCursor);
        laneEditor_->getInsertSectionHandle()->updatePos(parentRoadItem_, event->scenePos(), laneSection_->getSStart(), laneSection_->getSEnd());
        laneEditor_->getInsertSectionHandle()->show();
    }
    else if (tool == ODD::TLE_DEL)
    {
        setCursor(Qt::CrossCursor);
    }

    // Text //
    //
    getParentRoadItem()->getTextItem()->setVisible(true);
    getParentRoadItem()->getTextItem()->setPos(event->scenePos());

    // Parent //
    //
    SectionItem::hoverEnterEvent(event); // pass to baseclass
}

void
LaneSectionItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = laneEditor_->getCurrentTool();
    if (tool == ODD::TLE_SELECT)
    {
        // does nothing //
    }
    else if (tool == ODD::TLE_ADD)
    {
        laneEditor_->getInsertSectionHandle()->hide();
    }
	else if (tool == ODD::TLE_ADD_WIDTH)
	{
		laneEditor_->getAddWidthHandle()->hide();
	}
    else if (tool == ODD::TLE_DEL)
    {
        // does nothing //
    }

    setCursor(Qt::ArrowCursor);

    // Text //
    //
    getParentRoadItem()->getTextItem()->setVisible(false);

    // Parent //
    //
    SectionItem::hoverLeaveEvent(event); // pass to baseclass
}

void
LaneSectionItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = laneEditor_->getCurrentTool();
    if (tool == ODD::TEL_SELECT)
    {
        // does nothing //
    }
    else if (tool == ODD::TLE_ADD)
    {
        laneEditor_->getInsertSectionHandle()->updatePos(parentRoadItem_, event->scenePos(), laneSection_->getSStart(), laneSection_->getSEnd());
    }
    else if (tool == ODD::TLE_DEL)
    {
        // does nothing //
    }

    // Parent //
    //
    SectionItem::hoverMoveEvent(event);
}

//*************//
// Delete Item
//*************//

bool
LaneSectionItem::deleteRequest()
{
    if (removeSection())
    {
        return true;
    }

    return false;
}
