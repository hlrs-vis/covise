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

#include "junctionlanesectionitem.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/commands/lanesectioncommands.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/trackcommands.hpp"
#include "src/data/roadsystem/track/trackelement.hpp"
#include "src/data/roadsystem/roadsystem.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/junction/junctionlaneitem.hpp"
#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"
#include "src/graph/items/roadsystem/roadtextitem.hpp"
#include "src/graph/items/roadsystem/roaditem.hpp"
#include "src/graph/graphview.hpp"
#include "src/graph/topviewgraph.hpp"

// Editor //
//
#include "src/graph/editors/junctioneditor.hpp"

#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

JunctionLaneSectionItem::JunctionLaneSectionItem(JunctionEditor *junctionEditor, RoadItem *parentRoadItem, LaneSection *laneSection)
    : SectionItem(parentRoadItem, laneSection)
    , junctionEditor_(junctionEditor)
    , laneSection_(laneSection)
{
    init();
}

JunctionLaneSectionItem::~JunctionLaneSectionItem()
{
}

void
JunctionLaneSectionItem::init()
{

    // Selection/Hovering //
    //
    //	setAcceptHoverEvents(true);
    setSelectable();

    // SectionItems //
    //
    foreach (Lane *lane, laneSection_->getLanes())
    {
        if (lane->getId() != 0)
        {
            new JunctionLaneItem(this, lane);
        }
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
JunctionLaneSectionItem::updateObserver()
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
                    new JunctionLaneItem(this, lane);
                }
            }
        }
    }
}

//################//
// SLOTS          //
//################//

bool
JunctionLaneSectionItem::removeSection()
{
    RemoveLaneSectionCommand *command = new RemoveLaneSectionCommand(laneSection_, NULL);
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

void
JunctionLaneSectionItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = junctionEditor_->getCurrentTool();
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
JunctionLaneSectionItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = junctionEditor_->getCurrentTool();
    if (tool == ODD::TJE_CREATE_LANE)
    {
        // parent: selection //
        //SectionItem::mouseReleaseEvent(event); // do not pass selection to baseclass, we select individual lanes, not laneSections
        Lane *lane = dynamic_cast<Lane *>(getLaneSection()->getSelectedChildElements().first());
        junctionEditor_->registerLane(lane);
    }
    else if (tool == ODD::TJE_CREATE_ROAD)
    {
        // parent: selection //
        //SectionItem::mouseReleaseEvent(event); // do not pass selection to baseclass, we select individual lanes, not laneSections
        RSystemElementRoad *road = laneSection_->getParentRoad();
        junctionEditor_->registerRoad(road);
    }
    else if (tool == ODD::TJE_SPLIT)
    {
        RSystemElementRoad *road = laneSection_->getParentRoad();
        double s = road->getSFromGlobalPoint(event->scenePos(), laneSection_->getSStart(), laneSection_->getSEnd());

        // Split Track and Road//
        //
        SplitTrackRoadCommand *command = new SplitTrackRoadCommand(road, s, NULL);
        getProjectGraph()->executeCommand(command);

        return;
    }

    else if (tool == ODD::TLE_ADD && (event->button() == Qt::LeftButton))
    {
        //if(laneSection_->getDegree() <= 1) // only lines can be split TODO find out, whether all lane widths are linear
        {
            // Split Section //
            //
            RSystemElementRoad *road = laneSection_->getParentRoad();
            double s = road->getSFromGlobalPoint(event->pos(), laneSection_->getSStart(), laneSection_->getSEnd());

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
JunctionLaneSectionItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = junctionEditor_->getCurrentTool();
    if (tool == ODD::TEL_SELECT)
    {
        // parent: selection //
        SectionItem::mouseMoveEvent(event); // pass to baseclass
    }
}

void
JunctionLaneSectionItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = junctionEditor_->getCurrentTool();
    if (tool == ODD::TLE_SELECT)
    {
        setCursor(Qt::PointingHandCursor);
    }
    else if (tool == ODD::TLE_ADD)
    {
        setCursor(Qt::CrossCursor);
        //		junctionEditor_->getInsertSectionHandle()->show();
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
JunctionLaneSectionItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = junctionEditor_->getCurrentTool();
    if (tool == ODD::TEL_SELECT)
    {
        // does nothing //
    }
    else if (tool == ODD::TLE_ADD)
    {
        //		junctionEditor_->getInsertSectionHandle()->hide();
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
JunctionLaneSectionItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    ODD::ToolId tool = junctionEditor_->getCurrentTool();
    if (tool == ODD::TEL_SELECT)
    {
        // does nothing //
    }
    else if (tool == ODD::TLE_ADD)
    {
        //		junctionEditor_->getInsertSectionHandle()->updatePos(parentRoadItem_, event->scenePos(), laneSection_->getSStart(), laneSection_->getSEnd());
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
JunctionLaneSectionItem::deleteRequest()
{
    if (removeSection())
    {
        return true;
    }

    return false;
}
