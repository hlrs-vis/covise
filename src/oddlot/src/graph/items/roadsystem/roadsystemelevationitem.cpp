/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   21.06.2010
**
**************************************************************************/

#include "roadsystemelevationitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"

// Items //
//

RoadSystemElevationItem::RoadSystemElevationItem(ProjectGraph *projectGraph, RoadSystem *roadSystem)
    : GraphElement(NULL, roadSystem)
    , projectGraph_(projectGraph)
    , roadSystem_(roadSystem)
{
    init();
}

RoadSystemElevationItem::~RoadSystemElevationItem()
{
    roadSystem_->detachObserver(this);
}

void
RoadSystemElevationItem::init()
{
    // Observer Pattern //
    //
    roadSystem_->attachObserver(this);

    // Selection/Highlighting //
    //
    setOpacitySettings(1.0, 1.0); // always highlighted
}

//##################//
// Observer Pattern //
//##################//

void
RoadSystemElevationItem::updateObserver()
{

    // Get change flags //
    //
    int changes = roadSystem_->getRoadSystemChanges();

    // Road //
    //
    if (changes & RoadSystem::CRS_RoadChange)
    {
        // A road has been added.
        //
        //		foreach(RoadMap * map, roadSystem_->getRoadMaps())
        //		{
        //			if(!mapItems_.contains(map->getId()))
        //			{
        //				// New Item //
        //				//
        //				RoadMapItem * mapItem = new RoadMapItem(this, map);
        //				mapItems_.insert(mapItem->getMap()->getId(), mapItem);
        //			}
        //
        //		}

        // A road has been deleted.
        //
        //		foreach(RoadMapItem * mapItem, mapItems_)
        //		{
        //			if(mapItem->getMap()->getDataElementChanges() & DataElement::CDE_DataElementDeleted
        //				|| mapItem->getMap()->getDataElementChanges() & DataElement::CDE_DataElementRemoved)
        //			{
        //				mapItems_.remove(mapItem->getMap()->getId());
        //				getProjectGraph()->addToGarbage(mapItem);
        //			}
        //		}
    }

    // Parent //
    //
    GraphElement::updateObserver();
}
