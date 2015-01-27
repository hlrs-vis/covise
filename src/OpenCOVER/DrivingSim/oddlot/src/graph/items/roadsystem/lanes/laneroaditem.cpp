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

#include "laneroaditem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"

// Items //
//
#include "laneroadsystemitem.hpp"
#include "lanesectionitem.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"

// Editor //
//
#include "src/graph/editors/laneeditor.hpp"

LaneRoadItem::LaneRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
{
    init();
}

LaneRoadItem::~LaneRoadItem()
{
}

void
LaneRoadItem::init()
{
    // ElevationEditor //
    //
    laneEditor_ = dynamic_cast<LaneEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());
    if (!laneEditor_)
    {
        qDebug("Warning 1006241105! ElevationRoadItem not created by an ElevationEditor");
    }
    // SectionItems //
    //
    foreach (LaneSection *section, getRoad()->getLaneSections())
    {
        new LaneSectionItem(laneEditor_, this, section);
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
LaneRoadItem::updateObserver()
{
    // Parent //
    //
    RoadItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Road //
    //
    int changes = getRoad()->getRoadChanges();

    if (changes & RSystemElementRoad::CRD_LaneSectionChange)
    {
        // A section has been added.
        //
        foreach (LaneSection *section, getRoad()->getLaneSections())
        {
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new LaneSectionItem(laneEditor_, this, section);
            }
        }
    }
}
