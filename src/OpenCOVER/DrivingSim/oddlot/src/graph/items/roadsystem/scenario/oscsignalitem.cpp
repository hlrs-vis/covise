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

#include "oscsignalitem.hpp"
#include "oscroaditem.hpp"

#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphview.hpp"
#include "src/graph/items/roadsystem/signal/signaltextitem.hpp"
#include "src/graph/items/roadsystem/roadsystemitem.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QKeyEvent>

OSCSignalItem::OSCSignalItem(RoadSystemItem *roadSystemItem, Signal *signal, QPointF pos)
    : GraphElement(roadSystemItem, signal)
	, roadSystemItem_(roadSystemItem)
    , signal_(signal)
    , pos_(pos)
{
    init();
}

OSCSignalItem::~OSCSignalItem()
{
}

void
OSCSignalItem::init()
{
    // Hover Events //
    //
    setAcceptHoverEvents(true);
 //   setSelectable();
	setFlag(ItemIsFocusable);

    // Context Menu //
    //

    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
        signalTextItem_ = new SignalTextItem(this, signal_);
        signalTextItem_->setZValue(1.0); // stack before siblings
    }


	road_ = signal_->getParentRoad(); 
    pos_ = road_->getGlobalPoint(signal_->getSStart(), signal_->getT());

	updateColor();
    updatePosition();

}


/*! \brief Sets the color according to the number of links.
*/
void
OSCSignalItem::updateColor()
{
	outerColor_.setRgb(80, 80, 80);
}

/*!
* Initializes the path (only once).
*/
void
OSCSignalItem::createPath()
{
    QPainterPath path;

    // Stopp line
    //
    if (signal_->getType() == "294")
    {
		setPen(QPen(outerColor_, 2, Qt::SolidLine, Qt::FlatCap, Qt::RoundJoin));

        LaneSection *laneSection = road_->getLaneSection(signal_->getSStart());
		QPointF normal = road_->getGlobalNormal(signal_->getSStart()).toPointF();
		QPointF pos = pos_ + (normal * signal_->getT());

        if (signal_->getValidFromLane() >= 0)
        {
            double width = laneSection->getLaneSpanWidth(0, signal_->getValidFromLane(), signal_->getSStart()) + road_->getLaneOffset(signal_->getSStart());
            path.moveTo(pos - width * normal);
		}
		else
        {
            double width = laneSection->getLaneSpanWidth(0, signal_->getValidFromLane() + 1, signal_->getSStart()) + road_->getLaneOffset(signal_->getSStart());
			path.moveTo(pos + width * normal);
		}
		if (signal_->getValidToLane() > 0)
		{
            double width = laneSection->getLaneSpanWidth(0, signal_->getValidToLane() - 1, signal_->getSStart()) + road_->getLaneOffset(signal_->getSStart());
            path.lineTo(pos - width * normal);
        }
        else
        {
            double width = laneSection->getLaneSpanWidth(0, signal_->getValidToLane(), signal_->getSStart()) + road_->getLaneOffset(signal_->getSStart());
			path.lineTo(pos + width * normal);
        }
    }
    else if (signal_->getType() == "293")
    {
        setPen(QPen(outerColor_, 0.2, Qt::SolidLine, Qt::FlatCap, Qt::RoundJoin));

        LaneSection *laneSection = road_->getLaneSection(signal_->getSStart());
		QPointF normal = road_->getGlobalNormal(signal_->getSStart()).toPointF();
		QPointF tangent = road_->getGlobalTangent(signal_->getSStart()).toPointF();
		QPointF pos = pos_ + (normal * signal_->getT());

        if (signal_->getValidFromLane() > 0)
        {
            double width = laneSection->getLaneSpanWidth(0, signal_->getValidFromLane(), signal_->getSStart()) + road_->getLaneOffset(signal_->getSStart());
            if (signal_->getValidToLane() >= 0)
            {
                while (width >= (laneSection->getLaneSpanWidth(0, signal_->getValidToLane(), signal_->getSStart()) + road_->getLaneOffset(signal_->getSStart())))
                {
					QPointF newPos = pos - width * normal;
                    path.moveTo(newPos);
                    path.lineTo(newPos + signal_->getValue() * tangent);
                    width -= 1;
                }
            }
            else
            {
                while (width >= (-laneSection->getLaneSpanWidth(0, signal_->getValidToLane(), signal_->getSStart()) + road_->getLaneOffset(signal_->getSStart())))
                {
					QPointF newPos = pos - width * normal;
                    path.moveTo(newPos);
                    path.lineTo(newPos + signal_->getValue() * tangent);
                    width -= 1;
                }
            }
        }
        else
        {
            double width = laneSection->getLaneSpanWidth(0, signal_->getValidFromLane(), signal_->getSStart()) + road_->getLaneOffset(signal_->getSStart());
            while (width <= (laneSection->getLaneSpanWidth(0, signal_->getValidToLane(), signal_->getSStart())) + road_->getLaneOffset(signal_->getSStart()))
            {
				QPointF newPos = pos + width * normal;
                path.moveTo(newPos);
                path.lineTo(newPos + signal_->getValue() * tangent);
                width += 1;
            }
        }
    }
	else
	{

		double length = 2.0;
		setBrush(QBrush(outerColor_));
		setPen(QPen(outerColor_));

		path.addEllipse(pos_, 4.0, 4.0);

		setPen(QPen(QColor(255, 255, 255)));
		path.moveTo(pos_.x() - length, pos_.y());
		path.lineTo(pos_.x() + length, pos_.y());

		path.moveTo(pos_.x(), pos_.y() - length);
		path.lineTo(pos_.x(), pos_.y() + length);
	}

	setPath(path);
    
}

/*
* Update position
*/
void
OSCSignalItem::updatePosition()
{

    pos_ = road_->getGlobalPoint(signal_->getSStart(), signal_->getT());

    createPath();
}


//################//
// EVENTS         //
//################//

void
OSCSignalItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{

	setCursor(Qt::OpenHandCursor);
	setFocus();

	// Text //
	//
	getSignalTextItem()->setVisible(true);
	getSignalTextItem()->setPos(event->scenePos());

	// Parent //
	//
	GraphElement::hoverEnterEvent(event); // pass to baseclass
}

void
OSCSignalItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);

    // Text //
    //
    getSignalTextItem()->setVisible(false);

    // Parent //
    //
    GraphElement::hoverLeaveEvent(event); // pass to baseclass
}

void
OSCSignalItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{

    // Parent //
    //
    GraphElement::hoverMoveEvent(event);
}
