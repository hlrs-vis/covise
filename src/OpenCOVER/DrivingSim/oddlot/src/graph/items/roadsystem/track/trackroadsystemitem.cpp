/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   05.07.2010
**
**************************************************************************/

#include "trackroadsystemitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/track/trackroaditem.hpp"

//################//
// CONSTRUCTOR    //
//################//

TrackRoadSystemItem::TrackRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem)
    : RoadSystemItem(topviewGraph, roadSystem)
{
    init();
}

TrackRoadSystemItem::~TrackRoadSystemItem()
{
}

void
TrackRoadSystemItem::init()
{
    foreach (RSystemElementRoad *road, getRoadSystem()->getRoads())
    {
        trackRoadItems_.insert(road, new TrackRoadItem(this, road));
    }
}

void
TrackRoadSystemItem::notifyDeletion()
{
    foreach (TrackRoadItem *trackRoadItem, trackRoadItems_)
    {
        trackRoadItem->notifyDeletion();
    }
}

void
TrackRoadSystemItem::addRoadItem(TrackRoadItem *item)
{
    trackRoadItems_.insert(item->getRoad(), item);
}

int
TrackRoadSystemItem::removeRoadItem(TrackRoadItem *item)
{
    return trackRoadItems_.remove(item->getRoad());
}

//##################//
// Handles          //
//##################//

/*! \brief .
*
*/
void
TrackRoadSystemItem::rebuildMoveRotateHandles()
{
    foreach (TrackRoadItem *trackRoadItem, trackRoadItems_)
    {
        trackRoadItem->rebuildMoveRotateHandles(true);
    }
}

/*! \brief .
*
*/
void
TrackRoadSystemItem::rebuildAddHandles()
{
    foreach (TrackRoadItem *trackRoadItem, trackRoadItems_)
    {
        trackRoadItem->rebuildAddHandles(true);
    }
}

/*! \brief .
*
*/
void
TrackRoadSystemItem::rebuildRoadMoveRotateHandles()
{
    foreach (TrackRoadItem *trackRoadItem, trackRoadItems_)
    {
        trackRoadItem->rebuildRoadMoveRotateHandles(true);
    }
}

/*! \brief .
*
*/
void
TrackRoadSystemItem::deleteHandles()
{
    foreach (TrackRoadItem *trackRoadItem, trackRoadItems_)
    {
        trackRoadItem->deleteHandles();
    }
}

//##################//
// Observer Pattern //
//##################//

void
TrackRoadSystemItem::updateObserver()
{
    // Parent //
    //
    RoadSystemItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Get change flags //
    //
    int changes = getRoadSystem()->getRoadSystemChanges();

    // Road //
    //
    if (changes & RoadSystem::CRS_RoadChange)
    {
        // A road has been added (or deleted - but that will be handled by the road item itself).
        //
        foreach (RSystemElementRoad *road, getRoadSystem()->getRoads())
        {
            if ((road->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (road->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                trackRoadItems_.insert(road, new TrackRoadItem(this, road));
            }
        }
    }
}
