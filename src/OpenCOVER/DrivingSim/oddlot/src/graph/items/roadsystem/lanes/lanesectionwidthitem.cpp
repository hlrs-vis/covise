/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/18/2010
**
**************************************************************************/

#include "lanesectionwidthitem.hpp"
#include "lanewidthitem.hpp"
#include "src/graph/items/roadsystem/lanes/lanewidthmovehandle.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/graph/items/roadsystem/lanes/lanewidthroadsystemitem.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/roadcommands.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"
#include "src/graph/profilegraph.hpp"
#include "lanesectionitem.hpp"
#include "lanelinkitem.hpp"

// Editor //
//
#include "src/graph/editors/laneeditor.hpp"

// Utils //
//
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>

#include <QMessageBox>

#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

LaneSectionWidthItem::LaneSectionWidthItem(LaneWidthRoadSystemItem *parent, Lane *lane)
    : GraphElement(parent, lane->getParentLaneSection())
    , parentLaneSection_(lane->getParentLaneSection())
    , lane_(lane)
    , moveHandles_(NULL)
{
    grandparentRoad_ = parentLaneSection_->getParentRoad();
    laneEditor_ = dynamic_cast<LaneEditor *>(getProfileGraph()->getProjectWidget()->getProjectEditor());
    if (!laneEditor_)
    {
        qDebug("Warning 1006241105! LaneSectionWidthItem not created by an LaneEditor");
    }

    init();
}

LaneSectionWidthItem::~LaneSectionWidthItem()
{
    // Observer Pattern //
    //
    grandparentRoad_->detachObserver(this);
    lane_->detachObserver(this);
    //foreach(moveHandles_
}

void
LaneSectionWidthItem::init()
{
    // Observer Pattern //
    //
    grandparentRoad_->attachObserver(this);
    lane_->attachObserver(this);

    // Selection/Hovering //
    //
    setAcceptHoverEvents(true);
    setSelectable();

    setOpacitySettings(1.0, 1.0); // ...always highlighted

    // SectionItems //
    //
    foreach (LaneWidth *section, lane_->getWidthEntries())
    {
        new LaneWidthItem(this, section);
    }
    // Color & Path //
    //
    updateColor();
    createPath();

    /*	todo create height sections foreach(ElevationSection * section, getRoad()->getElevationSections())
	{
		// SectionItem //
		//
		new ElevationSectionPolynomialItem(this, section);
	}*/

    // MoveHandles //
    //
    moveHandles_ = new QGraphicsPathItem(this);
    rebuildMoveHandles();

    // ContextMenu //
    //
}

//################//
// SLOTS          //
//################//

//################//
// TOOLS          //
//################//

//################//
// GRAPHICS       //
//################//

/*! \brief Sets the color according to ...
*/

void
LaneSectionWidthItem::updateColor()
{
    QPen pen;
    pen.setWidth(1);
    pen.setCosmetic(true); // constant size independent of scaling
    pen.setColor(ODD::instance()->colors()->darkGreen());
    pen.setStyle(Qt::DashDotLine);

    setPen(pen);
}

void
LaneSectionWidthItem::createPath()
{
    // Initialization //
    //
    double sStart = parentLaneSection_->getSStart();
    double sEnd = parentLaneSection_->getSEnd();
    if (sEnd < sStart)
        sEnd = sStart;

    QVector<QPointF> points(2);
    points[0] = QPointF(sStart - (sEnd - sStart) / 100.0, 0);
    points[1] = QPointF(sEnd + (sEnd - sStart) / 100.0, 0);

    // Psycho-Path //
    //
    QPainterPath path;
    path.addPolygon(QPolygonF(points));

    setPath(path);
}

void
LaneSectionWidthItem::rebuildMoveHandles()
{
    deleteMoveHandles();

    //
    LaneWidth *previousSection = NULL;
    LaneWidthMoveHandle *previousHandle = NULL;
    foreach (LaneWidth *widthPoint, getLane()->getWidthEntries())
    {
        LaneWidthMoveHandle *handle = new LaneWidthMoveHandle(laneEditor_, moveHandles_);
        handle->registerHighSlot(widthPoint);
        if (previousSection)
        {
            handle->registerLowSlot(previousSection);
        }

        // Handle Degrees of freedom //
        //
        // NOTE: do not confuse DOF degree with the polynomial's degree

        handle->setDOF(2);

        // Done //
        //
        previousSection = widthPoint;
        previousHandle = handle;
    }
    if (previousHandle)
        previousHandle->setDOF(2);

    // Last handle of the road //
    //
    if (previousSection)
    {
        LaneWidthMoveHandle *handle = new LaneWidthMoveHandle(laneEditor_, moveHandles_);
        handle->registerLowSlot(previousSection);
        handle->setDOF(2);
    }

    // Z depth //
    //
    moveHandles_->setZValue(1.0);
}

void
LaneSectionWidthItem::deleteMoveHandles()
{
    foreach (QGraphicsItem *child, moveHandles_->childItems())
    {
        laneEditor_->getProfileGraph()->addToGarbage(child);
    }
}

//################//
// OBSERVER       //
//################//

void
LaneSectionWidthItem::updateObserver()
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
    if ((changes & Lane::CLN_TypeChanged))
    {
        updateColor();
    }

    if (changes & Lane::CLN_WidthsChanged)
    {
        foreach (LaneWidth *section, getLane()->getWidthEntries())
        {
            bool added = false;
            if ((section->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (section->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new LaneWidthItem(this, section);
                added = true;
            }
            if (added)
            {
                rebuildMoveHandles();
            }
        }
    }
}

//################//
// EVENTS         //
//################//

//void
//	LaneSectionWidthItem
//	::mousePressEvent(QGraphicsSceneMouseEvent * event)
//{
//}

void
LaneSectionWidthItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    //parentLaneSectionItem_->mouseReleaseEvent(event);
}

//void
//	LaneSectionWidthItem
//	::mouseMoveEvent(QGraphicsSceneMouseEvent * event)
//{

//}
