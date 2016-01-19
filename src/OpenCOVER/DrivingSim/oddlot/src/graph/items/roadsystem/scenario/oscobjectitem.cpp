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

#include "oscobjectitem.hpp"

#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"
#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Widget //
//
#include "src/gui/projectwidget.hpp"

// Graph //
//

#include "src/graph/items/roadsystem/signal/objecttextitem.hpp"
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
#include <QKeyEvent>

OSCObjectItem::OSCObjectItem(RoadSystemItem *roadSystemItem, Object *object, QPointF pos)
    : GraphElement(roadSystemItem, object)
	, roadSystemItem_(roadSystemItem)
    , object_(object)
    , pos_(pos)
	, path_(NULL)
{
    init();
}

OSCObjectItem::~OSCObjectItem()
{
}

void
OSCObjectItem::init()
{
    // Hover Events //
    //
    setAcceptHoverEvents(true);
    setSelectable();
	setFlag(ItemIsFocusable);


    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
        objectTextItem_ = new ObjectTextItem(this, object_);
        objectTextItem_->setZValue(1.0); // stack before siblings
    }

	road_ = object_->getParentRoad(); 
	pos_ = road_->getGlobalPoint(object_->getSStart(), object_->getT());

    updatePosition();

	doPan_ = false;
	copyPan_ = false;
}


/*! \brief Sets the color according to the number of links.
*/
void
OSCObjectItem::updateColor()
{
   outerColor_.setRgb(80, 80, 80);
}

/*!
* Initializes the path (only once).
*/
void
OSCObjectItem::createPath()
{
	if (path_)
	{
		delete path_;
	}

	path_ = new QPainterPath();

    setBrush(QBrush(outerColor_));
    setPen(QPen(outerColor_));

	double t = object_->getT();
	double w;
	if (object_->getT() <= 0)
	{
		w = object_->getWidth();
	}
	else
	{
		w = -object_->getWidth();
	}


	if (object_->getRepeatLength() > NUMERICAL_ZERO3) // Object is repeated
	{
		double currentS = object_->getRepeatS();

		double totalLength = 0.0;

		double dist;
		if (object_->getRepeatDistance() > 0.0)
		{
			dist = object_->getRepeatDistance();
		}
		else
		{
			//				double dist = 4; // TODO get configured tesselation length Jutta knows where to get this from
			dist = 1 / getProjectGraph()->getProjectWidget()->getLODSettings()->TopViewEditorPointsPerMeter;
		}

		LaneSection * currentLaneSection = road_->getLaneSection(currentS);
		double sSection = currentS - currentLaneSection->getSStart();
		currentLaneSection = NULL;

		int currentLaneId = 0;
		double d = 0.0;
		while ((totalLength < object_->getRepeatLength()) && (currentS < road_->getLength()))
		{

			if (road_->getLaneSection(currentS) != currentLaneSection)
			{
				LaneSection * newLaneSection = road_->getLaneSection(currentS);
				while (currentLaneSection && (currentLaneSection != newLaneSection))
				{
					if (object_->getT() < -NUMERICAL_ZERO3)
					{
						t = -currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentLaneSection->getSEnd()) + d;
					}
					else if (object_->getT() > NUMERICAL_ZERO3)
					{
						t = currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentLaneSection->getSEnd()) + d;
					}

					currentLaneSection = road_->getLaneSectionNext(currentLaneSection->getSStart() + NUMERICAL_ZERO3);
					currentLaneId = currentLaneSection->getLaneId(0, t);
					sSection = 0;
				}

				currentLaneSection = newLaneSection;
				currentLaneId = currentLaneSection->getLaneId(sSection, t);
				if (object_->getT() < -NUMERICAL_ZERO3) 
				{
					if (fabs(t)  < currentLaneSection->getLaneSpanWidth( 0, currentLaneId + 1, currentS) + currentLaneSection->getLaneWidth(currentLaneId, currentS)/2)
					{
						currentLaneId++;
					}
					d = currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + t;
				}
				else if (object_->getT() > NUMERICAL_ZERO3) 
				{
					if (t < currentLaneSection->getLaneSpanWidth( 0, currentLaneId - 1, currentS) + currentLaneSection->getLaneWidth(currentLaneId, currentS)/2)
					{
						currentLaneId--;
					}
					d = t  -  currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS);
				}
			}

			if (object_->getT() < -NUMERICAL_ZERO3)
			{
				t = -currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + d;
			}
			else if (object_->getT() > NUMERICAL_ZERO3)
			{
				t = currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + d;
			}
			QPointF currentPos = road_->getGlobalPoint(currentS, t);



			if (object_->getRepeatDistance() > 0.0) // multiple objects
			{
				double length = object_->getRadius() / 4;

				if (object_->getRadius() > 0.0) // circular object
				{
					path_->addEllipse(currentPos, object_->getRadius(), object_->getRadius());

					//               setPen(QPen(QColor(255, 255, 255)));
					path_->moveTo(currentPos.x() - length, currentPos.y());
					path_->lineTo(currentPos.x() + length, currentPos.y());

					path_->moveTo(currentPos.x(), currentPos.y() - length);
					path_->lineTo(currentPos.x(), currentPos.y() + length);
				}
				else
				{
					QMatrix transformationMatrix;
					QMatrix rotationMatrix;
					QPainterPath tmpPath;

					transformationMatrix.translate(currentPos.x(), currentPos.y());
					tmpPath.addRect(w / -2.0, object_->getLength() / -2.0, w / 2.0, object_->getLength() / 2.0);
					rotationMatrix.rotate(road_->getGlobalHeading(currentS) - 90 + object_->getHeading());
					tmpPath = transformationMatrix.map(rotationMatrix.map(tmpPath));
					*path_ += tmpPath;
				}

				if ((totalLength + dist) > object_->getRepeatLength())
					dist = object_->getRepeatLength() - totalLength;

			}
			else
			{

				// line object
				if (totalLength == 0)
				{
					path_->moveTo(currentPos.x(), currentPos.y());
				}
				else
				{
					path_->lineTo(currentPos.x(), currentPos.y());
					path_->moveTo(currentPos.x(), currentPos.y());
				}


				if ((totalLength + dist) > object_->getRepeatLength())
				{
					QPointF currentPos = road_->getGlobalPoint(currentS + totalLength - object_->getRepeatLength(), t);
					path_->lineTo(currentPos.x(), currentPos.y());
				}
			}

			totalLength += dist;
			currentS += dist;

		}
	}
	else
	{
		if (object_->getRadius() > 0.0) // circular object
		{
			path_->addEllipse(pos_, object_->getRadius(), object_->getRadius());
			double length = object_->getRadius() / 4;

			//            setPen(QPen(QColor(255, 255, 255)));
			path_->moveTo(pos_.x() - length, pos_.y());
			path_->lineTo(pos_.x() + length, pos_.y());

			path_->moveTo(pos_.x(), pos_.y() - length);
			path_->lineTo(pos_.x(), pos_.y() + length);
		}
		else
		{
			QMatrix transformationMatrix;
			QMatrix rotationMatrix;

			transformationMatrix.translate(pos_.x(), pos_.y());

			path_->addRect(w / -2.0, object_->getLength() / -2.0, w / 2.0, object_->getLength() / 2.0);
			rotationMatrix.rotate(road_->getGlobalHeading(object_->getSStart()) - 90 + object_->getHeading());
			*path_ = transformationMatrix.map(rotationMatrix.map(*path_));
		}
	}

	setPath(*path_);
}

/*
* Update position
*/
void
OSCObjectItem::updatePosition()
{

    pos_ = road_->getGlobalPoint(object_->getSStart(), object_->getT());

    createPath();
}

//*************//
// Delete Item
//*************//

bool
OSCObjectItem::deleteRequest()
{
    if (removeObject())
    {
        return true;
    }

    return false;
}

//################//
// SLOTS          //
//################//

bool
OSCObjectItem::removeObject()
{
    RemoveObjectCommand *command = new RemoveObjectCommand(object_, road_);
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

void
OSCObjectItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
	setCursor(Qt::OpenHandCursor);
	setFocus();

    // Text //
    //
    getObjectTextItem()->setVisible(true);
    getObjectTextItem()->setPos(event->scenePos());

    // Parent //
    //
    //GraphElement::hoverEnterEvent(event); // pass to baseclass
}

void
OSCObjectItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
	setCursor(Qt::ArrowCursor);
	if (!copyPan_)
	{
		clearFocus();
	}

    // Text //
    //
    getObjectTextItem()->setVisible(false);

    // Parent //
    //
    //GraphElement::hoverLeaveEvent(event); // pass to baseclass
}

void
OSCObjectItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{

    // Parent //
    //
    //GraphElement::hoverMoveEvent(event);
}

