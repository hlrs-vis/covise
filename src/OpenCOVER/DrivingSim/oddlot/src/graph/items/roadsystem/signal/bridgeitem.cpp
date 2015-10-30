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
    , bridge_(bridge)
    , pos_(pos)
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

    updateColor();
    updatePosition();
    createPath();
}

/*! \brief Sets the color according to the number of links.
*/
void
BridgeItem::updateColor()
{
	outerColor_.setHsv((categorySize_ - 1) * 360/(categorySize_ + 1), 255, 255, 255);
}

/*!
* Initializes the path (only once).
*/
void
BridgeItem::createPath()
{
    setBrush(QBrush(outerColor_));
    setPen(QPen(outerColor_, 2.0));

    QPainterPath path;

    RSystemElementRoad *road = bridge_->getParentRoad();

    if (bridge_->getLength() > NUMERICAL_ZERO3) // Bridge is repeated
    {
        double totalLength = 0.0;
        double currentS = bridge_->getSStart();

        // Left and right side //
        //
        while ((totalLength < bridge_->getLength()) && (currentS < road->getLength()))
        {
            LaneSection *laneSection = bridge_->getParentRoad()->getLaneSection(currentS);
            double t = laneSection->getLaneSpanWidth(0, laneSection->getRightmostLaneId() - 1, currentS);
            QPointF currentPos = bridge_->getParentRoad()->getGlobalPoint(currentS, t);

            if (totalLength == 0)
            {
                path.moveTo(currentPos.x(), currentPos.y());
            }
            else
            {
                path.lineTo(currentPos.x(), currentPos.y());
                path.moveTo(currentPos.x(), currentPos.y());
            }

            //				double dist = 4; // TODO get configured tesselation length Jutta knows where to get this from
            double dist = 1 / getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;

            if ((totalLength + dist) > bridge_->getLength())
            {
                QPointF currentPos = bridge_->getParentRoad()->getGlobalPoint(currentS + (bridge_->getLength() - totalLength), t);
                path.lineTo(currentPos.x(), currentPos.y());
            }

            totalLength += dist;
            currentS += dist;
        }

        totalLength = 0.0;
        currentS = bridge_->getSStart();

        // Left and right side //
        //
        while ((totalLength < bridge_->getLength()) && (currentS < road->getLength()))
        {
            LaneSection *laneSection = bridge_->getParentRoad()->getLaneSection(currentS);
            double t = -laneSection->getLaneSpanWidth(laneSection->getLeftmostLaneId(), 0, currentS);
            QPointF currentPos = bridge_->getParentRoad()->getGlobalPoint(currentS, t);

            if (totalLength == 0)
            {
                path.moveTo(currentPos.x(), currentPos.y());
            }
            else
            {
                path.lineTo(currentPos.x(), currentPos.y());
                path.moveTo(currentPos.x(), currentPos.y());
            }

            //				double dist = 4; // TODO get configured tesselation length Jutta knows where to get this from
            double dist = 1 / getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;

            if ((totalLength + dist) > bridge_->getLength())
            {
                QPointF currentPos = bridge_->getParentRoad()->getGlobalPoint(currentS + (bridge_->getLength() - totalLength), t);
                path.lineTo(currentPos.x(), currentPos.y());
            }

            totalLength += dist;
            currentS += dist;
        }
    }

    setPath(path);
}

/*
* Update position
*/
void
BridgeItem::updatePosition()
{

    //	pos_ = bridge_->getParentRoad()->getGlobalPoint(bridge_->getSStart(), bridge_->getT());
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
    RemoveBridgeCommand *command = new RemoveBridgeCommand(bridge_, bridge_->getParentRoad());
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

void
BridgeItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{

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
    ODD::ToolId tool = signalEditor_->getCurrentTool(); // Editor Delete Bridge
    if (tool == ODD::TSG_DEL)
    {
        removeBridge();
    }
    else
    {
        GraphElement::mousePressEvent(event); // pass to baseclass
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
