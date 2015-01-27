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

#include "junctionroadsystemitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/junction/junctionroaditem.hpp"

//################//
// CONSTRUCTOR    //
//################//

JunctionRoadSystemItem::JunctionRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem)
    : RoadSystemItem(topviewGraph, roadSystem)
{
    init();
}

JunctionRoadSystemItem::~JunctionRoadSystemItem()
{
}

void
JunctionRoadSystemItem::init()
{
    foreach (RSystemElementRoad *road, getRoadSystem()->getRoads())
    {
        junctionRoadItems_.insert(road, new JunctionRoadItem(this, road));
    }
}

void
JunctionRoadSystemItem::notifyDeletion()
{
    foreach (JunctionRoadItem *junctionRoadItem, junctionRoadItems_)
    {
        junctionRoadItem->notifyDeletion();
    }
}

void
JunctionRoadSystemItem::addRoadItem(JunctionRoadItem *item)
{
    junctionRoadItems_.insert(item->getRoad(), item);
}

int
JunctionRoadSystemItem::removeRoadItem(JunctionRoadItem *item)
{
    return junctionRoadItems_.remove(item->getRoad());
}

//##################//
// Handles          //
//##################//

/*! \brief .
*
*/
void
JunctionRoadSystemItem::rebuildMoveHandles()
{
    foreach (JunctionRoadItem *junctionRoadItem, junctionRoadItems_)
    {
        junctionRoadItem->rebuildMoveHandles();
    }
}

/*! \brief .
*
*/
void
JunctionRoadSystemItem::rebuildAddHandles()
{
    foreach (JunctionRoadItem *junctionRoadItem, junctionRoadItems_)
    {
        junctionRoadItem->rebuildAddHandles();
    }
}

/*! \brief .
*
*/
void
JunctionRoadSystemItem::deleteHandles()
{
    foreach (JunctionRoadItem *junctionRoadItem, junctionRoadItems_)
    {
        junctionRoadItem->deleteHandles();
    }
}

//##################//
// Observer Pattern //
//##################//

void
JunctionRoadSystemItem::updateObserver()
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
                junctionRoadItems_.insert(road, new JunctionRoadItem(this, road));
            }
        }
    }
}
