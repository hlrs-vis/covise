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

#include "objectitem.hpp"
#include "signalroaditem.hpp"

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
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"

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

ObjectItem::ObjectItem(RoadSystemItem *roadSystemItem, Object *object, QPointF pos)
    : GraphElement(roadSystemItem, object)
	, roadSystemItem_(roadSystemItem)
    , object_(object)
    , pos_(pos)
	, path_(NULL)
{
    init();
}

ObjectItem::~ObjectItem()
{
}

void
ObjectItem::init()
{
    // Hover Events //
    //
    setAcceptHoverEvents(true);
    setSelectable();
	setFlag(ItemIsFocusable);

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

    QAction *removeRoadAction = getRemoveMenu()->addAction(tr("Object"));
    connect(removeRoadAction, SIGNAL(triggered()), this, SLOT(removeObject()));

    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
        objectTextItem_ = new ObjectTextItem(this, object_);
        objectTextItem_->setZValue(1.0); // stack before siblings
    }

	road_ = object_->getParentRoad(); 
	closestRoad_ = road_;
	pos_ = road_->getGlobalPoint(object_->getSStart(), object_->getT());

    updateCategory();
    updatePosition();

	doPan_ = false;
	copyPan_ = false;
}

void 
ObjectItem::updateCategory()
{
	ObjectContainer *objectContainer = signalManager_->getObjectContainer(object_->getType());
	if (objectContainer)
	{
		QString category = objectContainer->getObjectCategory();
		int i = 360 / (categorySize_ + 1);
		outerColor_.setHsv(signalManager_->getCategoryNumber(category) * i, 255, 255, 255);
	}
	else
	{
		outerColor_.setRgb(80, 80, 80);
	}
}

/*! \brief Sets the color according to the number of links.
*/
void
ObjectItem::updateColor()
{
  //  outerColor_.setRgb(255, 0, 255);
}

/*!
* Initializes the path (only once).
*/
void
ObjectItem::createPath()
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
						t = -currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentLaneSection->getSEnd()) + road_->getLaneOffset(currentLaneSection->getSEnd()) + d;
					}
					else if (object_->getT() > NUMERICAL_ZERO3)
					{
						t = currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentLaneSection->getSEnd()) + road_->getLaneOffset(currentLaneSection->getSEnd()) + d;
					}

					currentLaneSection = road_->getLaneSectionNext(currentLaneSection->getSStart() + NUMERICAL_ZERO3);
					currentLaneId = currentLaneSection->getLaneId(0, t);
					sSection = 0;
				}

				currentLaneSection = newLaneSection;
				currentLaneId = currentLaneSection->getLaneId(sSection, t);
				if (object_->getT() < -NUMERICAL_ZERO3) 
				{
					if (fabs(t)  < currentLaneSection->getLaneSpanWidth( 0, currentLaneId + 1, currentS) + road_->getLaneOffset(currentS) + currentLaneSection->getLaneWidth(currentLaneId, currentS)/2)
					{
						currentLaneId++;
					}
					d = currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + t + road_->getLaneOffset(currentS);
				}
				else if (object_->getT() > NUMERICAL_ZERO3) 
				{
					if (t < currentLaneSection->getLaneSpanWidth( 0, currentLaneId - 1, currentS) + road_->getLaneOffset(currentS) + currentLaneSection->getLaneWidth(currentLaneId, currentS)/2)
					{
						currentLaneId--;
					}
					d = t  -  currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + road_->getLaneOffset(currentS);
				}
			}

			if (object_->getT() < -NUMERICAL_ZERO3)
			{
				t = -currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + road_->getLaneOffset(currentS) + d;
			}
			else if (object_->getT() > NUMERICAL_ZERO3)
			{
				t = currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + road_->getLaneOffset(currentS) + d;
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

			path_->addRect(w / -2.0, 0, w / 2.0, object_->getLength());
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
ObjectItem::updatePosition()
{

    pos_ = road_->getGlobalPoint(object_->getSStart(), object_->getT());

    createPath();
}

/* 
* Duplicate item
*/
void
	ObjectItem::duplicate()
{
	Object * newObject = object_->getClone();
	AddObjectCommand *command = new AddObjectCommand(newObject, object_->getParentRoad(), NULL);
	getProjectGraph()->executeCommand(command);
}

/* 
* Move item 
*/
void
ObjectItem::move(QPointF &diff)
{
	path_->translate(diff);
	setPath(*path_);
}

//*************//
// Delete Item
//*************//

bool
ObjectItem::deleteRequest()
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
ObjectItem::removeObject()
{
    RemoveObjectCommand *command = new RemoveObjectCommand(object_, road_);
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

void
ObjectItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
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
ObjectItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
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
ObjectItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{

    // Parent //
    //
    //GraphElement::hoverMoveEvent(event);
}

void
ObjectItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	pressPos_ = lastPos_ = event->scenePos();
	closestRoad_ = road_;
    ODD::ToolId tool = signalEditor_->getCurrentTool(); // Editor Delete Object
    if (tool == ODD::TSG_DEL)
    {
        removeObject();
    }
    else
    {
		doPan_ = true;

		if (copyPan_)
		{
			signalEditor_->duplicate();
		}

        GraphElement::mousePressEvent(event); // pass to baseclass
    }
}

void
ObjectItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{	
	if (doPan_)
	{

		QPointF newPos = event->scenePos();
		QPointF p = newPos - lastPos_;
		signalEditor_->move(p);
		lastPos_ = newPos;

		double s;
		QVector2D vec;
		double dist;

		RSystemElementRoad * nearestRoad = getProjectData()->getRoadSystem()->findClosestRoad( newPos, s, dist, vec);
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
ObjectItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	GraphElement::mouseReleaseEvent(event);

	double diff = (lastPos_ - pressPos_).manhattanLength();
	if (diff > 0.01) // otherwise item has not been moved by intention
	{
		if (doPan_)
		{
			if (object_->getRepeatLength() > NUMERICAL_ZERO3) // Object is repeated
			{
				pos_ = road_->getGlobalPoint(object_->getRepeatS(), object_->getT()) + lastPos_ - pressPos_; //??
			}
			else
			{
				pos_ = road_->getGlobalPoint(object_->getSStart(), object_->getT()) + lastPos_ - pressPos_;
			}
			QPointF dist = lastPos_ - pressPos_;
			signalEditor_->translate(dist);

		}
	}
	else
	{
		pos_ = lastPos_;
	}

	doPan_ = false;
}

/*! \brief Key events for panning, etc.
*
*/
void
ObjectItem::keyPressEvent(QKeyEvent *event)
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
ObjectItem::keyReleaseEvent(QKeyEvent *event)
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
ObjectItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Object //
    //
    int changes = object_->getObjectChanges();

	if ((changes & Object::CEL_TypeChange))
    {
        updateCategory();
        updatePosition();
    }
    else if ((changes & Object::CEL_ParameterChange))
    {
        updatePosition();
    }
}
