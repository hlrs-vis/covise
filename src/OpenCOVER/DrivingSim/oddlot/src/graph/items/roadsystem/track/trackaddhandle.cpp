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

TrackAddHandle::TrackAddHandle(TrackEditor *trackEditor, QGraphicsItem *parentItem, RSystemElementRoad *road, bool isStart)
    : LinkHandle(parentItem)
    , trackEditor_(trackEditor)
    , road_(road)
    , isStart_(isStart)
{
    // Flags //
    //
    //setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    //setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

    // Observer Pattern //
    //
    road_->attachObserver(this);

    if (isStart_)
    {
        track_ = road_->getTrackComponent(0.0);
    }
    else
    {
        track_ = road->getTrackComponent(road->getLength());
    }
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

    if (isStart_)
    {
        setPos(road_->getGlobalPoint(0.0));
        setRotation(road_->getGlobalHeading(0.0));
    }
    else
    {
        setPos(road_->getGlobalPoint(road_->getLength()));
        setRotation(road_->getGlobalHeading(road_->getLength()));
    }
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
    // Get change flags //
    //
    int changes = road_->getRoadChanges();

    if ((changes & RSystemElementRoad::CRD_TrackSectionChange)
        || (changes & RSystemElementRoad::CRD_LengthChange)
        || (changes & RSystemElementRoad::CRD_ShapeChange))
    {
        // Check if track is still the first/last one //
        //
        if (isStart_)
        {
            if (track_ != road_->getTrackComponent(0.0))
            {
                track_->detachObserver(this);
                track_ = road_->getTrackComponent(0.0);
                track_->attachObserver(this);
            }
        }
        else
        {
            if (track_ != road_->getTrackComponent(road_->getLength()))
            {
                track_->detachObserver(this);
                track_ = road_->getTrackComponent(road_->getLength());
                track_->attachObserver(this);
            }
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
