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

#include "oscitem.hpp"
#include "osctextitem.hpp"

#include "src/graph/items/roadsystem/scenario/oscroaditem.hpp"

#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/commands/osccommands.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphview.hpp"
//#include "src/graph/items/roadsystem/signal/signaltextitem.hpp"
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "src/graph/editors/osceditor.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"

// OpenScenario //
//
#include "oscVehicle.h"
#include "oscObject.h"
#include "oscMember.h"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QKeyEvent>

OSCItem::OSCItem(RoadSystemItem *roadSystemItem, OpenScenario::oscObject *oscObject, OpenScenario::oscObjectBase *catalogElement, const QPointF &pos)
    : GraphElement(roadSystemItem, NULL)
	, roadSystemItem_(roadSystemItem)
    , oscObject_(oscObject)
	, selectedObject_(catalogElement)
	, path_(NULL)
	, pos_(pos)
{

    init();
}

OSCItem::~OSCItem()
{
}

/*!
* Initializes the path 
*/
QPainterPath *
	createVehiclePath(OpenScenario::oscObjectBase *vehicle, const QPointF &pos)
{
	QPainterPath *path = new QPainterPath();

	oscIntValue *iv = dynamic_cast<oscIntValue *>(vehicle->getMember("vehicleClass")->getGenerateValue());
	if (iv)
	{
		switch (iv->getValue())
		{
		case oscVehicle::car:
			{
				QPolygonF polygon;
				polygon << QPointF(-5,-2) << QPointF(-5,0) << QPointF(-5,0) << QPointF(-1.3,2) << QPointF(1.3,2) << QPointF(2.7,0) << QPointF(4.2,0) << QPointF(4.8,-0.8) << QPointF(5,-2);
				path->addPolygon(polygon);
				path->closeSubpath();
				path->addEllipse(QPointF(-3,-2.1), 0.8, 0.8);
				path->addEllipse(QPointF(3,-2.1), 0.8, 0.8);
				path->translate(pos);
				break;
			}
		case oscVehicle::van:
			{
				break;
			}
		default:
			{
				path->addRect(pos.x(), pos.y(), 10, 10);
			}

	/*	truck,
		trailer,
		bus,
		motorbike,
		bicycle,
		train,
		tram,*/
		}
	}

	return path;

}

void
OSCItem::init()
{
	
    // Hover Events //
    //
    setAcceptHoverEvents(true);
//    setSelectable();
	setFlag(QGraphicsItem::ItemIsSelectable);
	setFlag(ItemIsFocusable);

    // OpenScenario Editor
    //
    oscEditor_ = dynamic_cast<OpenScenarioEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());

    // Context Menu //
    //

    QAction *removeElementAction = getRemoveMenu()->addAction(tr("OpenScenario Object"));
    connect(removeElementAction, SIGNAL(triggered()), this, SLOT(removeElement()));

    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
        oscTextItem_ = new OSCTextItem(this, oscObject_, pos_);
        oscTextItem_->setZValue(1.0); // stack before siblings
    }

	OpenScenario::oscObjectBase *oscPosition = oscObject_->getMember("initPosition")->getObject();
	OpenScenario::oscObjectBase *oscPosRoad = oscPosition->getMember("positionRoad")->getGenerateObject();
	roadID_ = QString::fromStdString(dynamic_cast<oscStringValue *>(oscPosRoad->getMember("roadId")->getValue())->getValue());
	road_ = roadSystemItem_->getRoadSystem()->getRoad(roadID_);
	closestRoad_ = road_;

	doPan_ = false;
	copyPan_ = false;

	const std::string typeName = selectedObject_->getOwnMember()->getTypeName();
	if (typeName == "oscVehicle")
	{
		createPath = createVehiclePath;
	}

	updateColor(typeName);
    updatePosition();
}


/*! \brief Sets the color according to the number of links.
*/
void
OSCItem::updateColor(const std::string &type)
{
	if (type == "oscVehicle")
	{
		color_ = Qt::black;
	}

	
//	setBrush(QBrush(color_));
	setPen(QPen(color_));
}


/*
* Update position
*/
void
OSCItem::updatePosition()
{


	QPainterPath *path = createPath(selectedObject_, pos_);
	setPath(*path);
}

//*************//
// Delete Item
//*************//

bool
OSCItem::deleteRequest()
{
    if (removeElement())
    {
        return true;
    }

    return false;
}

//################//
// SLOTS          //
//################//

bool
OSCItem::removeElement()
{
 /*   RemoveSignalCommand *command = new RemoveSignalCommand(signal_, road_);
    return getProjectGraph()->executeCommand(command); */
	return false;
}

//################//
// EVENTS         //
//################//

void
OSCItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{

	setCursor(Qt::OpenHandCursor);
	setFocus();

	// Text //
	//
	oscTextItem_->setVisible(true);
	oscTextItem_->setPos(event->scenePos());

	// Parent //
	//
	GraphElement::hoverEnterEvent(event); // pass to baseclass
}

void
OSCItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
	if (!copyPan_)
	{
		clearFocus();
	}

    // Text //
    //
    oscTextItem_->setVisible(false);

    // Parent //
    //
    GraphElement::hoverLeaveEvent(event); // pass to baseclass
}

void
OSCItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{

    // Parent //
    //
    GraphElement::hoverMoveEvent(event);
}

void
OSCItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    pressPos_ = lastPos_ = event->scenePos();
    ODD::ToolId tool = oscEditor_->getCurrentTool(); // Editor Delete Signal
    if (tool == ODD::TSG_DEL)
    {
        removeElement();
    }
    else 
    {

		doPan_ = true;
		if (copyPan_)
		{
	/*		Signal * newSignal = signal_->getClone();
			AddSignalCommand *command = new AddSignalCommand(newSignal, signal_->getParentRoad(), NULL);
			getProjectGraph()->executeCommand(command); */
		}
        GraphElement::mousePressEvent(event); // pass to baseclass

    }
}

void
OSCItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{	
	if (doPan_)
	{

		QPointF newPos = event->scenePos();
		pos_ += newPos - lastPos_;
		lastPos_ = newPos;
	//	createPath();

	//	QPointF to = road_->getGlobalPoint(signal_->getSStart(), signal_->getT()) + lastPos_ - pressPos_;
		QPointF to;

		double s;
		QVector2D vec;
		double dist;

		RSystemElementRoad * nearestRoad = oscEditor_->findClosestRoad( to, s, dist, vec);
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
OSCItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	GraphElement::mouseReleaseEvent(event);

    if (doPan_)
    {
		double diff = (lastPos_ - pressPos_).manhattanLength();
		if (diff > 0.01) // otherwise item has not been moved by intention
		{
	//		pos_ = road_->getGlobalPoint(signal_->getSStart(), signal_->getT()) + lastPos_ - pressPos_;
		/*	bool parentChanged = oscEditor_->translateObject(element_, closestRoad_, pos_);

			if (!parentChanged)
			{
				updatePosition();
			}*/
		}
		else
		{
			pos_ = lastPos_;
		}

		doPan_ = false;
    }
}

/*! \brief Key events for panning, etc.
*
*/
void
OSCItem::keyPressEvent(QKeyEvent *event)
{
    // TODO: This will not notice a key pressed, when the view is not active
    switch (event->key())
    {
	case Qt::Key_Shift:
        copyPan_ = true;
        break;

    default:
        QGraphicsItem::keyPressEvent(event);
    }
}

/*! \brief Key events for panning, etc.
*
*/
void
OSCItem::keyReleaseEvent(QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_Shift:
        copyPan_ = false;
		if (!isHovered())
		{
			clearFocus();
		}
        break;


    default:
        QGraphicsItem::keyReleaseEvent(event);
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
OSCItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Signal //
    //
 /*   int changes = signal_->getSignalChanges();

    if ((changes & Signal::CEL_TypeChange))
    {
        updatePosition();
    }
    else if ((changes & Signal::CEL_ParameterChange))
    {
        updatePosition();
    } */
}
