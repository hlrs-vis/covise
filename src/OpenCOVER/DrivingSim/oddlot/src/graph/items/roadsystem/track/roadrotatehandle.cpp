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

#include "roadrotatehandle.hpp"

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

RoadRotateHandle::RoadRotateHandle(TrackEditor *trackEditor, RSystemElementRoad *road, QGraphicsItem *parent)
    : RotateHandle(parent)
    , trackEditor_(trackEditor)
    , road_(road)
{
    // Flags //
    //
    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

    // Observer Pattern //
    //
    road_->attachObserver(this);

    if (road_)
        firstTrack_ = road_->getTrackComponent(0.0);
    if (firstTrack_)
        firstTrack_->attachObserver(this);

    // Color/Transform //
    //
    updateColor();
    updateTransform();
}

RoadRotateHandle::~RoadRotateHandle()
{
    if (road_)
        road_->detachObserver(this);
    if (firstTrack_)
        firstTrack_->detachObserver(this);
    if (trackEditor_)
        trackEditor_->unregisterRoadRotateHandle(this);
}

//################//
// FUNCTIONS      //
//################//

void
RoadRotateHandle::updateColor()
{
    setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
    setPen(QPen(ODD::instance()->colors()->darkGreen()));
}

void
RoadRotateHandle::updateTransform()
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
RoadRotateHandle::updateObserver()
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
RoadRotateHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //

    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        if (value.toBool())
        {
            trackEditor_->registerRoadRotateHandle(this);
        }
        else
        {
            trackEditor_->unregisterRoadRotateHandle(this);
        }
        return value;
    }

    return QGraphicsItem::itemChange(change, value);
}

void
RoadRotateHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	trackEditor_->setCacheMode(road_, TrackEditor::CacheMode::NoCache);
	RotateHandle::mousePressEvent(event);
}

void
RoadRotateHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    QPointF direction = event->scenePos() - scenePos();
    double heading = (atan2(direction.y(), direction.x()) * 360.0 / (2.0 * M_PI)) - road_->getGlobalHeading(0.0); // (catches x == 0)
    while (heading < 0.0)
    {
        heading += 360.0;
    }

    // Let the trackEditor handle the movement //
    //
    if (isSelected()) // only if this handle is actually selected
    {
        trackEditor_->rotateRoadRotateHandles(scenePos(), heading);
    }

    //	RotateHandle::mouseMoveEvent(event); // pass to baseclass
}

void
RoadRotateHandle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	trackEditor_->setCacheMode(road_, TrackEditor::CacheMode::DeviceCache);
	RotateHandle::mouseReleaseEvent(event);
}
