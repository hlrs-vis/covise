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

#include "signalroaditem.hpp"

#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"

#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/commands/signalcommands.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//

#include "src/graph/items/roadsystem/roadtextitem.hpp"
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "src/graph/items/roadsystem/signal/signalitem.hpp"
#include "src/graph/items/roadsystem/signal/objectitem.hpp"
#include "src/graph/items/roadsystem/signal/bridgeitem.hpp"
#include "src/graph/items/roadsystem/signal/signalhandle.hpp"
#include "src/graph/editors/signaleditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>

SignalRoadItem::SignalRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road)
    : RoadItem(roadSystemItem, road)
    , roadSystemItem_(roadSystemItem)
    , road_(road)
{
    init();
}

SignalRoadItem::~SignalRoadItem()
{
}

void
SignalRoadItem::init()
{
    // Hover Events //
    //
    setAcceptHoverEvents(true);
    signalEditor_ = dynamic_cast<SignalEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());

    foreach (Signal *signal, road_->getSignals())
    {
        new SignalItem(roadSystemItem_, signal, road_->getGlobalPoint(signal->getSStart(), signal->getT()));
    }

    foreach (Object *object, road_->getObjects())
    {
        new ObjectItem(roadSystemItem_, object, road_->getGlobalPoint(object->getSStart(), object->getT()));
    }

    foreach (Bridge *bridge, road_->getBridges())
    {
        new BridgeItem(roadSystemItem_, bridge, road_->getGlobalPoint(bridge->getSStart(), 0.0));
    }

    updateColor();
    createPath();
}
/*! \brief Sets the color according to the number of links.
*/
void
SignalRoadItem::updateColor()
{

    setBrush(QBrush(ODD::instance()->colors()->brightBlue()));
    setPen(QPen(ODD::instance()->colors()->darkBlue()));
}

void
SignalRoadItem::createPath()
{
    RSystemElementRoad *road = getRoad();

    // Initialization //
    //
    double sStart = 0.0;
    double sEnd = road->getLength();
    if (sEnd < sStart)
        sEnd = sStart;

    //	double pointsPerMeter = 1.0; // BAD: hard coded!
    double pointsPerMeter = getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;
    int pointCount = int(ceil((sEnd - sStart) * pointsPerMeter)); // TODO curvature...
    if (pointCount < 2)
    {
        pointCount = 2;
    }
    QVector<QPointF> points(2 * pointCount + 1);
    double segmentLength = (sEnd - sStart) / (pointCount - 1);

    // Right side //
    //
    for (int i = 0; i < pointCount; ++i)
    {
        double s = sStart + i * segmentLength; // [sStart, sEnd]
        points[i] = road->getGlobalPoint(s, road->getMinWidth(s));
    }

    // Left side //
    //
    for (int i = 0; i < pointCount; ++i)
    {
        double s = sEnd - i * segmentLength; // [sEnd, sStart]
        if (s < 0.0)
            s = 0.0; // can happen due to numerical inaccuracy (around -1.0e-15)
        points[i + pointCount] = road->getGlobalPoint(s, road->getMaxWidth(s));
    }

    // End point //
    //
    points[2 * pointCount] = road->getGlobalPoint(sStart, road->getMinWidth(sStart));

    // Psycho-Path //
    //
    QPainterPath path;
    path.addPolygon(QPolygonF(points));

    setPath(path);
}

//################//
// EVENTS         //
//################//

void
SignalRoadItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = signalEditor_->getCurrentTool();
    /*	if(tool == ODD::TRT_SELECT)
	{
		// Parent: selection //
		//
		return SectionItem::mousePressEvent(event);
	}
	else
	{
		return; // prevent selection by doing nothing
	}*/
}

void
SignalRoadItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = signalEditor_->getCurrentTool();
    /*	if(tool == ODD::TRT_SELECT)
	{
		// Parent: selection //
		//
		SectionItem::mouseReleaseEvent(event);
	}*/

    pos_ = event->scenePos();
    if ((tool == ODD::TSG_SIGNAL) || (tool == ODD::TSG_OBJECT) || (tool == ODD::TSG_BRIDGE))
    {
        // Add new Signal //
        //
        double s = road_->getSFromGlobalPoint(pos_, 0.0, road_->getLength());

        // calculate t at s
        //
        double t = road_->getTFromGlobalPoint(pos_, s);

        if (tool == ODD::TSG_SIGNAL)
        {
            int validToLane = 0;

            LaneSection *laneSection = road_->getLaneSection(s);
            if (t < 0)
            {
                validToLane = laneSection->getRightmostLaneId();
            }
            else
            {
                validToLane = laneSection->getLeftmostLaneId();
            }
            QList<UserData *> userData;
            Signal *newSignal = new Signal("signal", "", s, t, false, Signal::NEGATIVE_TRACK_DIRECTION, 0.0, "Germany", -1, "", -1, 0.0, 0.0, 0.0, 0.0, true, 2, 0, validToLane);
            AddSignalCommand *command = new AddSignalCommand(newSignal, road_, NULL);

            getProjectGraph()->executeCommand(command);
        }
        else if (tool == ODD::TSG_OBJECT)
        {
            Object *newObject = new Object("object", "", "", s, t, 0.0, 0.0, Object::NEGATIVE_TRACK_DIRECTION, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, false, s, 0.0, 0.0, "");
            AddObjectCommand *command = new AddObjectCommand(newObject, road_, NULL);

            getProjectGraph()->executeCommand(command);
        }
        else if (tool == ODD::TSG_BRIDGE)
        {
            QList<ObjectCorner *> corners;
            Bridge *newBridge = new Bridge("bridge", "", "", 0, s, 100.0);
            AddBridgeCommand *command = new AddBridgeCommand(newBridge, road_, NULL);

            getProjectGraph()->executeCommand(command);
        }
    }

    else if ((tool == ODD::TSG_OBJECT) || (tool == ODD::TSG_BRIDGE))
    {
        // Add new Object //
        //
        RSystemElementRoad *road = getRoad();
        double s = road->getSFromGlobalPoint(event->scenePos(), 0.0, road->getLength());
    }
}

void
SignalRoadItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = signalEditor_->getCurrentTool();
    if (tool == ODD::TSG_SELECT)
    {
        // Parent: selection //
        //
        //	SectionItem::mouseMoveEvent(event); // pass to baseclass
    }
    else if (tool == ODD::TSG_SIGNAL)
    {
        // does nothing //
    }
    else if (tool == ODD::TSG_OBJECT)
    {
        // does nothing //
    }
}

void
SignalRoadItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    /*	ODD::ToolId tool = signalEditor_->getCurrentTool();
	if(tool == ODD::TSG_SELECT)
	{
		setCursor(Qt::PointingHandCursor);
	}
	else if((tool == ODD::TSG_SIGNAL) || (tool == ODD::TSG_OBJECT))
	{
		setCursor(Qt::CrossCursor);
		signalEditor_->getInsertSignalHandle()->show();
	}
	else if(tool == ODD::TSG_DEL)
	{
		setCursor(Qt::CrossCursor);
	}

	// Text //
	//
	getTextItem()->setVisible(true);
	getTextItem()->setPos(event->scenePos());

	// Parent //
	//
	RoadItem::hoverEnterEvent(event); // pass to baseclass

	*/
}

void
SignalRoadItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    /*	ODD::ToolId tool = signalEditor_->getCurrentTool();
	if(tool == ODD::TSG_SELECT)
	{
		// does nothing //
	}
	else if((tool == ODD::TSG_SIGNAL) || (tool == ODD::TSG_OBJECT))
	{
		signalEditor_->getInsertSignalHandle()->hide();
	}
	else if(tool == ODD::TSG_DEL)
	{
		// does nothing //
	}

	setCursor(Qt::ArrowCursor);

	// Text //
	//
	getTextItem()->setVisible(false);

	// Parent //
	//
	RoadItem::hoverLeaveEvent(event); // pass to baseclass

	*/
}

void
SignalRoadItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    /*	ODD::ToolId tool = signalEditor_->getCurrentTool();
	if(tool == ODD::TSG_SELECT)
	{
		// does nothing //
	}
	else if((tool == ODD::TSG_SIGNAL) || (tool == ODD::TSG_OBJECT))
	{
		signalEditor_->getInsertSignalHandle()->updatePos(this, event->scenePos(), 0.0, road_->getLength());
	}
	else if(tool == ODD::TSG_DEL)
	{
		// does nothing //
	}

	// Parent //
	//
	RoadItem::hoverMoveEvent(event);

	*/
}

//##################//
// Observer Pattern //
//##################//

void
SignalRoadItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Get change flags //
    //
    int changes = road_->getRoadChanges();

    if (changes & RSystemElementRoad::CRD_SignalChange)
    {
        // A signal has been added.
        //
        QMultiMap<double, Signal *>::const_iterator iter = road_->getSignals().constBegin();
        while (iter != road_->getSignals().constEnd())
        {
            Signal *element = iter.value();
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new SignalItem(roadSystemItem_, element, pos_);
            }

            iter++;
        }
    }
    else if (changes & RSystemElementRoad::CRD_ObjectChange)
    {
        // A object has been added.
        //
        QMultiMap<double, Object *>::const_iterator iter = road_->getObjects().constBegin();
        while (iter != road_->getObjects().constEnd())
        {
            Object *element = iter.value();
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new ObjectItem(roadSystemItem_, element, pos_);
            }

            iter++;
        }
    }
    else if (changes & RSystemElementRoad::CRD_BridgeChange)
    {
        // A bridge has been added.
        //
        QMultiMap<double, Bridge *>::const_iterator iter = road_->getBridges().constBegin();
        while (iter != road_->getBridges().constEnd())
        {
            Bridge *element = iter.value();
            if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
                || (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
            {
                new BridgeItem(roadSystemItem_, element, pos_);
            }

            iter++;
        }
    }
}
