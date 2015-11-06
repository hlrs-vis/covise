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

#include "bridgeitem.hpp"

#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/sections/bridgeobject.hpp"
#include "src/data/roadsystem/sections/tunnelobject.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//

#include "src/graph/items/roadsystem/signal/bridgetextitem.hpp"
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "src/graph/items/roadsystem/roaditem.hpp"
#include "src/graph/editors/signaleditor.hpp"

// Manager //
//
#include "src/data/signalmanager.hpp" 

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QMatrix>

BridgeItem::BridgeItem(RoadSystemItem *roadSystemItem, Bridge *bridge, QPointF pos)
    : GraphElement(roadSystemItem, bridge)
	, roadSystemItem_(roadSystemItem)
    , bridge_(bridge)
    , pos_(pos)
	, path_(NULL)
{
    init();
}

BridgeItem::~BridgeItem()
{
}

void
BridgeItem::init()
{
    // Hover Events //
    //
    setAcceptHoverEvents(true);
    setSelectable();

    // Signal Editor
    //
    signalEditor_ = dynamic_cast<SignalEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());

	// Signal Manager
	//
	signalManager_ = getProjectData()->getProjectWidget()->getMainWindow()->getSignalManager();

	// Category Size
	//
	categorySize_ = signalManager_->getCategoriesSize();

    // Context Menu //
    //

    QAction *removeRoadAction = getRemoveMenu()->addAction(tr("Bridge"));
    connect(removeRoadAction, SIGNAL(triggered()), this, SLOT(removeBridge()));

    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
        bridgeTextItem_ = new BridgeTextItem(this);
        bridgeTextItem_->setZValue(1.0); // stack before siblings
    }

	road_ = bridge_->getParentRoad(); 
	closestRoad_ = road_;

    updateColor();
    updatePosition();
    createPath();

	doPan_ = false;
}

/*! \brief Sets the color according to the number of links.
*/
void
BridgeItem::updateColor()
{
	if (dynamic_cast<Tunnel *>(bridge_)) // Bridge is a tunnel //
	{
		outerColor_.setHsv(categorySize_ * 360/(categorySize_ + 1), 255, 255, 255);
	}
	else
	{
		outerColor_.setHsv((categorySize_ - 1) * 360/(categorySize_ + 1), 255, 255, 255);
	}
}

/*!
* Initializes the path (only once).
*/
void
BridgeItem::createPath()
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
BridgeItem::updatePosition()
{

 //   pos_ = road_->getGlobalPoint(bridge_->getSStart());
    updateColor();
    createPath();
}

//*************//
// Delete Item
//*************//

bool
BridgeItem::deleteRequest()
{
    if (removeBridge())
    {
        return true;
    }

    return false;
}

//################//
// SLOTS          //
//################//

bool
BridgeItem::removeBridge()
{
    RemoveBridgeCommand *command = new RemoveBridgeCommand(bridge_, road_);
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

void
BridgeItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
	setCursor(Qt::OpenHandCursor);

    // Text //
    //
    getBridgeTextItem()->setVisible(true);
    getBridgeTextItem()->setPos(event->scenePos());

    // Parent //
    //
    //GraphElement::hoverEnterEvent(event); // pass to baseclass
}

void
BridgeItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
	setCursor(Qt::ArrowCursor);

    // Text //
    //
    getBridgeTextItem()->setVisible(false);

    // Parent //
    //
    //GraphElement::hoverLeaveEvent(event); // pass to baseclass
}

void
BridgeItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{

    // Parent //
    //
    //GraphElement::hoverMoveEvent(event);
}

void
BridgeItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	pressPos_ = lastPos_ = event->scenePos();
    ODD::ToolId tool = signalEditor_->getCurrentTool(); // Editor Delete Bridge
    if (tool == ODD::TSG_DEL)
    {
        removeBridge();
    }
    else
    {
		doPan_ = true;
        GraphElement::mousePressEvent(event); // pass to baseclass
    }
}

void
BridgeItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{	
	if (doPan_)
	{

		QPointF newPos = event->scenePos();
		path_->translate(newPos - lastPos_);
		lastPos_ = newPos;
		setPath(*path_);

		QPointF to = road_->getGlobalPoint(bridge_->getSStart()) + lastPos_ - pressPos_;

		double s;
		QVector2D vec;
		double dist;

		RSystemElementRoad * nearestRoad = signalEditor_->findClosestRoad( to, s, dist, vec);
		if (!nearestRoad)
		{
			nearestRoad = road_;
		}
		if (nearestRoad != closestRoad_)
		{
			RoadItem *nearestRoadItem = roadSystemItem_->getRoadItem(nearestRoad->getID());
			nearestRoadItem->setHighlighting(true);
			setZValue(nearestRoadItem->zValue() + 1);
			roadSystemItem_->getRoadItem(closestRoad_->getID())->setHighlighting(false);
			closestRoad_ = nearestRoad;
		}

		GraphElement::mouseMoveEvent(event);
	}
}

void
BridgeItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	GraphElement::mouseReleaseEvent(event);

    if (doPan_)
    {
		pos_ = road_->getGlobalPoint(bridge_->getSStart()) + lastPos_ - pressPos_;
		signalEditor_->translateBridge(bridge_, closestRoad_, pos_);

		doPan_ = false;
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
BridgeItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Bridge //
    //
    int changes = bridge_->getBridgeChanges();

    if ((changes & Bridge::CEL_ParameterChange))
    {
        updatePosition();
    }
}
