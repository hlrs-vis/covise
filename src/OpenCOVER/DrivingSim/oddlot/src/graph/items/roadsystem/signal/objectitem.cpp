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

#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"

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
#include "src/graph/editors/signaleditor.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QMatrix>

ObjectItem::ObjectItem(RoadSystemItem *roadSystemItem, Object *object, QPointF pos)
    : GraphElement(roadSystemItem, object)
    , object_(object)
    , pos_(pos)
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

    // Signal Editor
    //
    signalEditor_ = dynamic_cast<SignalEditor *>(getProjectGraph()->getProjectWidget()->getProjectEditor());

    // Context Menu //
    //

    QAction *removeRoadAction = getRemoveMenu()->addAction(tr("Object"));
    connect(removeRoadAction, SIGNAL(triggered()), this, SLOT(removeObject()));

    if (getTopviewGraph()) // not for profile graph
    {
        // Text //
        //
        objectTextItem_ = new ObjectTextItem(this);
        objectTextItem_->setZValue(1.0); // stack before siblings
    }

    updateColor();
    updatePosition();
    createPath();
}

/*! \brief Sets the color according to the number of links.
*/
void
ObjectItem::updateColor()
{
    outerColor_.setRgb(255, 0, 255);
}

/*!
* Initializes the path (only once).
*/
void
ObjectItem::createPath()
{
    setBrush(QBrush(outerColor_));
    setPen(QPen(outerColor_));

    QPainterPath path;

    RSystemElementRoad *road = object_->getParentRoad();

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

    LaneSection * currentLaneSection = road->getLaneSection(currentS);
    double sSection = currentS - currentLaneSection->getSStart();
    currentLaneSection = NULL;
    double t = object_->getT();
    int currentLaneId = 0;
    double d = 0.0;
    while ((totalLength < object_->getRepeatLength()) && (currentS < road->getLength()))
    {

        if (road->getLaneSection(currentS) != currentLaneSection)
        {
            LaneSection * newLaneSection = road->getLaneSection(currentS);
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

                currentLaneSection = road->getLaneSectionNext(currentLaneSection->getSStart() + NUMERICAL_ZERO3);
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
        QPointF currentPos = object_->getParentRoad()->getGlobalPoint(currentS, t);


        if (object_->getRepeatDistance() > 0.0) // multiple objects
        {
            double length = object_->getRadius() / 4;

            if (object_->getRadius() > 0.0) // circular object
            {
                path.addEllipse(currentPos, object_->getRadius(), object_->getRadius());

                setPen(QPen(QColor(255, 255, 255)));
                path.moveTo(currentPos.x() - length, currentPos.y());
                path.lineTo(currentPos.x() + length, currentPos.y());

                path.moveTo(currentPos.x(), currentPos.y() - length);
                path.lineTo(currentPos.x(), currentPos.y() + length);
            }
            else
            {
                QMatrix transformationMatrix;
                QMatrix rotationMatrix;
                QPainterPath tmpPath;
                transformationMatrix.translate(currentPos.x(), currentPos.y());
                tmpPath.addRect(object_->getWidth() / -2.0, object_->getLength() / -2.0, object_->getWidth() / 2.0, object_->getLength() / 2.0);
                rotationMatrix.rotate(road->getGlobalHeading(currentS) - 90 + object_->getHeading());
                tmpPath = transformationMatrix.map(rotationMatrix.map(tmpPath));
                path += tmpPath;
            }

            if ((totalLength + dist) > object_->getRepeatLength())
                dist = object_->getRepeatLength() - totalLength;

	    }
	    else
	    {

             // line object
                if (totalLength == 0)
                {
                    path.moveTo(currentPos.x(), currentPos.y());
                }
                else
                {
                    path.lineTo(currentPos.x(), currentPos.y());
                    path.moveTo(currentPos.x(), currentPos.y());
                }


                if ((totalLength + dist) > object_->getRepeatLength())
                {
                    QPointF currentPos = object_->getParentRoad()->getGlobalPoint(currentS + totalLength - object_->getRepeatLength(), t);
                    path.lineTo(currentPos.x(), currentPos.y());
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
            path.addEllipse(pos_, object_->getRadius(), object_->getRadius());
            double length = object_->getRadius() / 4;

            setPen(QPen(QColor(255, 255, 255)));
            path.moveTo(pos_.x() - length, pos_.y());
            path.lineTo(pos_.x() + length, pos_.y());

            path.moveTo(pos_.x(), pos_.y() - length);
            path.lineTo(pos_.x(), pos_.y() + length);
        }
        else
        {
            QMatrix transformationMatrix;
            QMatrix rotationMatrix;

            transformationMatrix.translate(pos_.x(), pos_.y());
            path.addRect(object_->getWidth() / -2.0, object_->getLength() / -2.0, object_->getWidth() / 2.0, object_->getLength() / 2.0);
            rotationMatrix.rotate(road->getGlobalHeading(object_->getSStart()) - 90 + object_->getHeading());
            path = transformationMatrix.map(rotationMatrix.map(path));
        }
    }

    setPath(path);
}

/*
* Update position
*/
void
ObjectItem::updatePosition()
{

    pos_ = object_->getParentRoad()->getGlobalPoint(object_->getSStart(), object_->getT());
    updateColor();
    createPath();
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
    RemoveObjectCommand *command = new RemoveObjectCommand(object_, object_->getParentRoad());
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

void
ObjectItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{

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
    ODD::ToolId tool = signalEditor_->getCurrentTool(); // Editor Delete Object
    if (tool == ODD::TSG_DEL)
    {
        removeObject();
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

    if ((changes & Object::CEL_ParameterChange))
    {
        updatePosition();
    }
}
