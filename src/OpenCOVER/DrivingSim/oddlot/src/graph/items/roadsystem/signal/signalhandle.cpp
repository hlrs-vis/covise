/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   18.03.2010
**
**************************************************************************/

#include "signalhandle.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "src/graph/items/roadsystem/roaditem.hpp"
#include "signalhandle.hpp"
#include "signalroaditem.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>

//################//
// CONSTRUCTOR    //
//################//

SignalHandle::SignalHandle(SignalRoadItem *parentSignalRoadItem)
    : QGraphicsPathItem(parentSignalRoadItem)
    , parentRoadSystemItem_(NULL)
    , parentSignalRoadItem_(parentSignalRoadItem)
    , moveItem_(false)
{
    // Mode 1: Parent is a SectionItem //
    //
    // Every section has one Section Handle that is used for moving the section.

    // Init //
    //
    init();

    // Flags //
    //
    setFlag(QGraphicsItem::ItemIsMovable, true);
    //setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    //setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
    //setAcceptHoverEvents(true);
}

SignalHandle::SignalHandle(RoadSystemItem *parentRoadSystemItem)
    : QGraphicsPathItem(parentRoadSystemItem)
    , parentRoadSystemItem_(parentRoadSystemItem)
    , parentSignalRoadItem_(NULL)
    , moveItem_(false)
{
    // Mode 2: Parent is a RoadSystemItem //
    //
    // The RoadSystemItem has one SignalHandle that is used for adding new sections.

    // Init //
    //
    init();
}

//################//
// FUNCTIONS      //
//################//

void
SignalHandle::init()
{
    // Create if necessary //
    //
    if (!SignalHandle::pathTemplate_)
    {
        SignalHandle::createPath();
    }

    // ZValue //
    //
    setZValue(1.0); // stack before siblings

    // Set path //
    //
    setPath(*pathTemplate_);
}

void
SignalHandle::updateTransform()
{
    bool tmp = moveItem_;
    moveItem_ = false; // set false so itemChange doesn't interrupt
    RSystemElementRoad *road = parentSignalRoadItem_->getRoad();
    setPos(road->getGlobalPoint(0.0));
    setRotation(road->getGlobalHeading(0.0));
    moveItem_ = tmp; // reset
}

void
SignalHandle::updatePos(SignalRoadItem *signalRoadItem, const QPointF &position, double sStartHint, double sEndHint)
{
    RSystemElementRoad *road = signalRoadItem->getRoad();

    // Calculate road coordinate //
    //
    double s = road->getSFromGlobalPoint(position, sStartHint, sEndHint);

    // Set Item Pose //
    //
    setPos(road->getGlobalPoint(s, 0.0));
    setRotation(road->getGlobalHeading(s));

    // Line that spans the whole road //
    //
    QPainterPath path;
    path.moveTo(0.0, road->getMinWidth(s));
    path.lineTo(0.0, road->getMaxWidth(s));
    path.addPath(*pathTemplate_);
    setPath(path);
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
SignalHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //

    // Mode 1 //
    //
    if (parentSignalRoadItem_)
    {
        if (change == QGraphicsItem::ItemPositionChange)
        {
            if (moveItem_)
            {
                RSystemElementRoad *road = parentSignalRoadItem_->getRoad();
                double s = road->getSFromGlobalPoint(value.toPointF(), -100.0, road->getLength());
                //				return parentSectionItem_->getRoadSection()->getParentRoad()->getGlobalPoint(s, 0.0);

                /*				MoveRoadSectionCommand * command = new MoveRoadSectionCommand(roadSection, s, parentSectionItem_->getRoadSectionType());
				if(command->isValid())
				{
					roadSection->getUndoStack()->push(command);
				}
				else
				{
					delete command;
				}*/

                return pos(); // no translation
            }
            else
            {
            }
        }
    }

    // Mode 2 //
    //
    //	else
    //	{
    //
    //	}

    return QGraphicsItem::itemChange(change, value);
}

void
SignalHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::ClosedHandCursor);
    moveItem_ = true;
    QGraphicsPathItem::mousePressEvent(event); // pass to baseclass
}

void
SignalHandle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::OpenHandCursor);
    moveItem_ = false;
    QGraphicsPathItem::mouseReleaseEvent(event); // pass to baseclass
}

//################//
// STATIC         //
//################//

/*!
* Initializes the path to NULL.
*/
QPainterPath *SignalHandle::pathTemplate_ = NULL;

/*!
* Initializes the path (only once).
*/
void
SignalHandle::createPath()
{
    // Check first //
    //
    if (SignalHandle::pathTemplate_)
    {
        return; // already created
    }

    // Create //
    //
    double size = 6.0;

    static QPainterPath pathTemplate; // deleted on application shutdown
    pathTemplate.moveTo(0.0, -size * 0.5);
    pathTemplate.lineTo(0.0, size * 0.5);
    pathTemplate.lineTo(0.0 + 0.25 * 1.414213 * size * 0.5, 0.0);
    pathTemplate.lineTo(0.0, -size * 0.5);

    SignalHandle::pathTemplate_ = &pathTemplate;
}
