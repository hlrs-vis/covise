/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.07.2010
**
**************************************************************************/

#include "elevationroadsystemitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/elevation/elevationroaditem.hpp"

// Editor //
//
//#include "src/graph/editors/elevationeditor.hpp"

//################//
// CONSTRUCTOR    //
//################//

ElevationRoadSystemItem::ElevationRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem)
    : RoadSystemItem(topviewGraph, roadSystem)
{
    init();
}

ElevationRoadSystemItem::ElevationRoadSystemItem(ProfileGraph *profileGraph, RoadSystem *roadSystem)
    : RoadSystemItem(profileGraph, roadSystem)
{
    init();
}

ElevationRoadSystemItem::~ElevationRoadSystemItem()
{
}

void
ElevationRoadSystemItem::init()
{
    foreach (RSystemElementRoad *road, getRoadSystem()->getRoads())
    {
        new ElevationRoadItem(this, road);
    }
}

//##################//
// Observer Pattern //
//##################//

void
ElevationRoadSystemItem::updateObserver()
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
                new ElevationRoadItem(this, road);
            }
        }
    }
}
