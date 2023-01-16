/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   10.05.2010
 **
 **************************************************************************/

#include "trackaddhandle.hpp"

 // Data //
 //
#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/editors/trackeditor.hpp"
#include "src/graph/items/roadsystem/track/trackcomponentitem.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QGraphicsSceneMouseEvent>

// Utils //
//
//#include "math.h"
#include "src/util/colorpalette.hpp"

//################//
// CONSTRUCTOR    //
//################//

TrackAddHandle::TrackAddHandle(TrackEditor *trackEditor, QGraphicsItem *parentItem, RSystemElementRoad *road, int laneId, double s, double t)
    : LinkHandle(parentItem)
    , trackEditor_(trackEditor)
    , road_(road)
    , laneId_(laneId)
    , s_(s)
    , t_(t)
{
    // Flags //
    //
    //setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    //setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

    // Observer Pattern //
    //
    road_->attachObserver(this);

    track_ = road_->getTrackComponent(s_);

    if (track_)
        track_->attachObserver(this);

    // Transformation //
    //
    updateTransformation();
    updateColor();
}

TrackAddHandle::~TrackAddHandle()
{

    // Observer Pattern //
    //
    if (road_)
        road_->detachObserver(this);
    if (track_)
        track_->detachObserver(this);

    trackEditor_->unregisterTrackAddHandle(this);
}

//################//
// FUNCTIONS      //
//################//

void
TrackAddHandle::updateTransformation()
{
    setHandleType(LinkHandle::DHLT_START);


    setPos(road_->getGlobalPoint(s_,t_));
    setRotation(road_->getGlobalHeading(s_));
}

QPointF 
TrackAddHandle::getPos()
{
    return road_->getGlobalPoint(s_, t_);
}

void
TrackAddHandle::updateColor()
{
    setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
    setPen(QPen(ODD::instance()->colors()->darkGreen()));
}

//################//
// OBSERVER       //
//################//

/*!
*
*/
void
TrackAddHandle::updateObserver()
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

    // Get change flags //
    //
    int changes = road_->getRoadChanges();

    if ((changes & RSystemElementRoad::CRD_TrackSectionChange)
        || (changes & RSystemElementRoad::CRD_LengthChange)
        || (changes & RSystemElementRoad::CRD_ShapeChange))
    {
        // Check if track is still the first/last one //
        //

        if (s_ < NUMERICAL_ZERO8)
        {
            if (track_ != road_->getTrackComponent(0.0))
            {
                track_->detachObserver(this);
                track_ = road_->getTrackComponent(0.0);
                track_->attachObserver(this);
            }
        }
        else if (track_ != road_->getTrackComponent(road_->getLength()))
        {
            track_->detachObserver(this);
            s_ = road_->getLength();
            track_ = road_->getTrackComponent(s_);
            track_->attachObserver(this);
        }

        // Update Transformation //
        //
        updateTransformation();
    }
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
TrackAddHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //
    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        if (value.toBool())
        {
            trackEditor_->registerTrackAddHandle(this);
        }
        else
        {
            trackEditor_->unregisterTrackAddHandle(this);
        }
        return value;
    }

    return QGraphicsItem::itemChange(change, value);
}

void
TrackAddHandle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::CrossCursor);
    Handle::hoverEnterEvent(event);
}

void
TrackAddHandle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
    Handle::hoverLeaveEvent(event);
}

