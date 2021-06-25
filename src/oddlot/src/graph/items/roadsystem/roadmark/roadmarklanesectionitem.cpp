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

#include "roadmarklanesectionitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"

// Graph //
//
#include "roadmarklaneitem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

//################//
// CONSTRUCTOR    //
//################//

RoadMarkLaneSectionItem::RoadMarkLaneSectionItem(RoadItem *parentRoadItem, LaneSection *laneSection)
    : SectionItem(parentRoadItem, laneSection)
    , laneSection_(laneSection)
{
    init();
}

RoadMarkLaneSectionItem::~RoadMarkLaneSectionItem()
{
}

void
RoadMarkLaneSectionItem::init()
{
    sectionHandle_->setVisible(false);

    // SectionItems //
    //
    foreach (Lane *lane, laneSection_->getLanes())
    {
        new RoadMarkLaneItem(this, lane);
    }

    setAcceptedMouseButtons(Qt::NoButton);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
RoadMarkLaneSectionItem::updateObserver()
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
                new RoadMarkLaneItem(this, lane);
            }
        }
    }
}
