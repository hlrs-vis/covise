/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   07.07.2010
**
**************************************************************************/

#include "roadmovehandle.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/roadsystem/track/trackcomponent.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"
#include "src/graph/editors/trackeditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneMouseEvent>

// Utils //
//
#include "math.h"
#include "src/util/colorpalette.hpp"

//################//
// CONSTRUCTOR    //
//################//

RoadMoveHandle::RoadMoveHandle(TrackEditor *trackEditor, RSystemElementRoad *road, QGraphicsItem *parent)
    : MoveHandle(parent)
    , trackEditor_(trackEditor)
    , road_(road)
{
    // Observer Pattern //
    //
    road_->attachObserver(this);

    firstTrack_ = road_->getTrackComponent(0.0);
    firstTrack_->attachObserver(this);

    // Color/Transform //
    //
    updateColor();
    updateTransform();
}

RoadMoveHandle::~RoadMoveHandle()
{
    road_->detachObserver(this);
    firstTrack_->detachObserver(this);
    trackEditor_->unregisterRoadMoveHandle(this);
}

//################//
// FUNCTIONS      //
//################//

void
RoadMoveHandle::updateColor()
{
    setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
    setPen(QPen(ODD::instance()->colors()->darkGreen()));
}

void
RoadMoveHandle::updateTransform()
{
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, false);
    setPos(road_->getGlobalPoint(0.0));
    setRotation(road_->getGlobalHeading(0.0));
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
}

//################//
// OBSERVER       //
//################//

/*!
*
* Does not know if highSlot or lowSlot has changed.
*/
void
RoadMoveHandle::updateObserver()
{
    int dataElementChanges = road_->getDataElementChanges();

    // Deletion //
    //
    if ((dataElementChanges & DataElement::CDE_DataElementDeleted)
        || (dataElementChanges & DataElement::CDE_DataElementRemoved))
    {
        trackEditor_->getProjectGraph()->addToGarbage(this);
        return;
    }

    // RoadChanges //
    //
    int roadChanges = road_->getRoadChanges();

    // Deletion //
    //
    if (roadChanges & RSystemElementRoad::CRD_TrackSectionChange)
    {
        if (firstTrack_ != road_->getTrackComponent(0.0))
        {
            firstTrack_->detachObserver(this);
            firstTrack_ = road_->getTrackComponent(0.0);
            firstTrack_->attachObserver(this);
        }
        updateTransform();
    }

    // TrackComponentChanges //
    //
    int trackChanges = firstTrack_->getTrackComponentChanges();

    if (trackChanges & TrackComponent::CTC_TransformChange)
    {
        updateTransform();
    }
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
RoadMoveHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //

    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        if (value.toBool())
        {
            trackEditor_->registerRoadMoveHandle(this);
        }
        else
        {
            trackEditor_->unregisterRoadMoveHandle(this);
        }
        return value;
    }

    return MoveHandle::itemChange(change, value);
}

//void
//	RoadMoveHandle
//	::mousePressEvent(QGraphicsSceneMouseEvent * event)
//{
//	setCursor(Qt::ClosedHandCursor);
//	MoveHandle::mousePressEvent(event); // pass to baseclass
//}
//
//void
//	RoadMoveHandle
//	::mouseReleaseEvent(QGraphicsSceneMouseEvent * event)
//{
//	setCursor(Qt::OpenHandCursor);
//	MoveHandle::mouseReleaseEvent(event); // pass to baseclass
//}

void
RoadMoveHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    // Let the trackEditor handle the movement //
    //
    if (isSelected()) // only if this handle is actually selected
    {
        trackEditor_->translateRoadMoveHandles(scenePos(), event->scenePos());
    }

    //	MoveHandle::mouseMoveEvent(event); // pass to baseclass
}

//void
//	RoadMoveHandle
//	::hoverEnterEvent(QGraphicsSceneHoverEvent * event)
//{
//	setCursor(Qt::OpenHandCursor);
//	MoveHandle::hoverEnterEvent(event);
//}
//
//void
//	RoadMoveHandle
//	::hoverLeaveEvent(QGraphicsSceneHoverEvent * event)
//{
//	setCursor(Qt::ArrowCursor);
//	MoveHandle::hoverLeaveEvent(event);
//}
//
//void
//	RoadMoveHandle
//	::hoverMoveEvent(QGraphicsSceneHoverEvent * event)
//{
//	MoveHandle::hoverMoveEvent(event);
//}
