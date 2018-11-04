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

#include "laneroadsystemitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/lanes/laneroaditem.hpp"

//################//
// CONSTRUCTOR    //
//################//

LaneRoadSystemItem::LaneRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem)
    : RoadSystemItem(topviewGraph, roadSystem)
{
    init();
}

LaneRoadSystemItem::~LaneRoadSystemItem()
{
}

void
LaneRoadSystemItem::init()
{
    foreach (RSystemElementRoad *road, getRoadSystem()->getRoads())
    {
        laneRoadItems_.insert(road, new LaneRoadItem(this, road));
    }
}

void
LaneRoadSystemItem::addRoadItem(LaneRoadItem *item)
{
	laneRoadItems_.insert(item->getRoad(), item);
}

int
LaneRoadSystemItem::removeRoadItem(LaneRoadItem *item)
{
	return laneRoadItems_.remove(item->getRoad());
}

LaneRoadItem *
LaneRoadSystemItem::getRoadItem(RSystemElementRoad *road)
{
	return laneRoadItems_.value(road, NULL);
}

//##################//
// Handles          //
//##################//

/*! \brief .
*
*/
void
LaneRoadSystemItem::rebuildMoveRotateHandles()
{
	foreach(LaneRoadItem *laneRoadItem, laneRoadItems_)
	{
		laneRoadItem->rebuildMoveRotateHandles(true);
	}
}

/*! \brief .
*
*/
void
LaneRoadSystemItem::deleteHandles()
{
	foreach(LaneRoadItem *laneRoadItem, laneRoadItems_)
	{
		laneRoadItem->deleteHandles();
	}
}

//##################//
// Observer Pattern //
//##################//

void
LaneRoadSystemItem::updateObserver()
{
    // Parent //
    //
    RoadSystemItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // RoadSystem //
    //
    int changes = getRoadSystem()->getRoadSystemChanges();

    if (changes & RoadSystem::CRS_RoadChange)
    {
        // A road has been added (or deleted - but that will be handled by the road item itself).
        //
        foreach (RSystemElementRoad *road, getRoadSystem()->getRoads())
        {
            if ((road->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (road->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                // SectionItem //
                //
				laneRoadItems_.insert(road, new LaneRoadItem(this, road));
            }
        }
    }
}
