/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#include "oscbridgeitem.hpp"

#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/sections/bridgeobject.hpp"
#include "src/data/roadsystem/sections/tunnelobject.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//

#include "src/graph/items/roadsystem/signal/bridgetextitem.hpp"
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "src/graph/items/roadsystem/roaditem.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QMatrix>

OSCBridgeItem::OSCBridgeItem(RoadSystemItem *roadSystemItem, Bridge *bridge, QPointF pos)
    : GraphElement(roadSystemItem, bridge)
	, roadSystemItem_(roadSystemItem)
    , bridge_(bridge)
    , pos_(pos)
	, path_(NULL)
{
    init();
}

OSCBridgeItem::~OSCBridgeItem()
{
}

void
OSCBridgeItem::init()
{
    // Hover Events //
    //
    setAcceptHoverEvents(true);
 //   setSelectable();
	setFlag(ItemIsFocusable);

	// Save a tunnel 
	//
	tunnel_ = dynamic_cast<Tunnel *>(bridge_);


    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
        bridgeTextItem_ = new BridgeTextItem(this, bridge_);
        bridgeTextItem_->setZValue(1.0); // stack before siblings
    }

	road_ = bridge_->getParentRoad(); 
	pos_ = road_->getGlobalPoint(bridge_->getSStart());

    updateColor();
    updatePosition();
    createPath();
}

/*! \brief Sets the color according to the number of links.
*/
void
OSCBridgeItem::updateColor()
{
	outerColor_.setRgb(80, 80, 80);
}

/*!
* Initializes the path (only once).
*/
void
OSCBridgeItem::createPath()
{
	if (path_)
	{
		delete path_;
	}

	path_ = new QPainterPath();

    setBrush(QBrush(outerColor_));
    setPen(QPen(outerColor_, 2.0));


    if (bridge_->getLength() > NUMERICAL_ZERO3) // Bridge is repeated
    {
        double totalLength = 0.0;
        double currentS = bridge_->getSStart();

        // Left and right side //
        //
        while ((totalLength < bridge_->getLength()) && (currentS < road_->getLength()))
        {
            LaneSection *laneSection = road_->getLaneSection(currentS);
            double t = laneSection->getLaneSpanWidth(0, laneSection->getRightmostLaneId() - 1, currentS);
            QPointF currentPos = road_->getGlobalPoint(currentS, t);

            if (totalLength == 0)
            {
                path_->moveTo(currentPos.x(), currentPos.y());
            }
            else
            {
                path_->lineTo(currentPos.x(), currentPos.y());
                path_->moveTo(currentPos.x(), currentPos.y());
            }

            //				double dist = 4; // TODO get configured tesselation length Jutta knows where to get this from
            double dist = 1 / getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;

            if ((totalLength + dist) > bridge_->getLength())
            {
                QPointF currentPos = road_->getGlobalPoint(currentS + (bridge_->getLength() - totalLength), t);
                path_->lineTo(currentPos.x(), currentPos.y());
            }

            totalLength += dist;
            currentS += dist;
        }

        totalLength = 0.0;
        currentS = bridge_->getSStart();

        // Left and right side //
        //
        while ((totalLength < bridge_->getLength()) && (currentS < road_->getLength()))
        {
            LaneSection *laneSection = road_->getLaneSection(currentS);
            double t = -laneSection->getLaneSpanWidth(laneSection->getLeftmostLaneId(), 0, currentS);
            QPointF currentPos = road_->getGlobalPoint(currentS, t);

            if (totalLength == 0)
            {
                path_->moveTo(currentPos.x(), currentPos.y());
            }
            else
            {
                path_->lineTo(currentPos.x(), currentPos.y());
                path_->moveTo(currentPos.x(), currentPos.y());
            }

            //				double dist = 4; // TODO get configured tesselation length Jutta knows where to get this from
            double dist = 1 / getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;

            if ((totalLength + dist) > bridge_->getLength())
            {
                QPointF currentPos = road_->getGlobalPoint(currentS + (bridge_->getLength() - totalLength), t);
                path_->lineTo(currentPos.x(), currentPos.y());
            }

            totalLength += dist;
            currentS += dist;
        }
    }

    setPath(*path_);
}

/*
* Update position
*/
void
OSCBridgeItem::updatePosition()
{

 //   pos_ = road_->getGlobalPoint(bridge_->getSStart());
    updateColor();
    createPath();
}


//################//
// EVENTS         //
//################//

void
OSCBridgeItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
	setCursor(Qt::OpenHandCursor);
	setFocus();

    // Text //
    //
    getBridgeTextItem()->setVisible(true);
    getBridgeTextItem()->setPos(event->scenePos());

    // Parent //
    //
    //GraphElement::hoverEnterEvent(event); // pass to baseclass
}

void
OSCBridgeItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
	setCursor(Qt::ArrowCursor);

    // Text //
    //
    getBridgeTextItem()->setVisible(false);

    // Parent //
    //
    //GraphElement::hoverLeaveEvent(event); // pass to baseclass
}


