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

#include "lanewidthroadsystemitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/elevation/elevationroaditem.hpp"

// Editor //
//
//#include "src/graph/editors/elevationeditor.hpp"

#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

//################//
// CONSTRUCTOR    //
//################//

LaneWidthRoadSystemItem::LaneWidthRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem)
    : RoadSystemItem(topviewGraph, roadSystem)
{
    init();
}

LaneWidthRoadSystemItem::LaneWidthRoadSystemItem(ProfileGraph *profileGraph, RoadSystem *roadSystem)
    : RoadSystemItem(profileGraph, roadSystem)
{
    init();
}

LaneWidthRoadSystemItem::~LaneWidthRoadSystemItem()
{
}

void
LaneWidthRoadSystemItem::init()
{
    settings = NULL;
    //setAcceptHoverEvents(true);
    setOpacitySettings(1.0, 1.0); // ...always highlighted
}

void LaneWidthRoadSystemItem::setSettings(LaneSettings *s)
{
    settings = s;
}

void LaneWidthRoadSystemItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    if (settings)
    {
        LaneWidth *laneWidth = settings->getLane()->getWidthEntry(0);
        //settings->getSectionHandle()->updatePos(this,event->scenePos(),laneWidth->getSSectionStart(),laneWidth->getSSectionEnd());
    }
}
void LaneWidthRoadSystemItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
}
void LaneWidthRoadSystemItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    if (settings)
    {
        LaneWidth *laneWidth = settings->getLane()->getWidthEntry(0);
        settings->getSectionHandle()->updatePos(this->getRoadItem(odrID::invalidID()), event->scenePos(), laneWidth->getSSectionStart(), laneWidth->getSSectionEnd());
    }
}

//##################//
// Observer Pattern //
//##################//

void
LaneWidthRoadSystemItem::updateObserver()
{
    // Parent //
    //
    RoadSystemItem::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }
}
