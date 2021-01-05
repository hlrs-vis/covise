/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/22/2010
**
**************************************************************************/

#include "roadmarklaneitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"

// Graph //
//
#include "roadmarklanesectionitem.hpp"
#include "roadmarkitem.hpp"

//################//
// CONSTRUCTOR    //
//################//

RoadMarkLaneItem::RoadMarkLaneItem(RoadMarkLaneSectionItem *parentLaneSectionItem, Lane *lane)
    : GraphElement(parentLaneSectionItem, lane)
    , lane_(lane)
{
    init();
}

RoadMarkLaneItem::~RoadMarkLaneItem()
{
}

void
RoadMarkLaneItem::init()
{
    foreach (LaneRoadMark *child, lane_->getRoadMarkEntries())
    {
        new RoadMarkItem(this, child);
    }

    setAcceptedMouseButtons(Qt::NoButton);
}

//################//
// OBSERVER       //
//################//

void
RoadMarkLaneItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Lane //
    //
    int changes = lane_->getLaneChanges();

    if (changes & Lane::CLN_RoadMarksChanged)
    {
        // A LaneRoadMark has been added.
        //
        foreach (LaneRoadMark *child, lane_->getRoadMarkEntries())
        {
            if ((child->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (child->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new RoadMarkItem(this, child);
            }
        }
    }
}
