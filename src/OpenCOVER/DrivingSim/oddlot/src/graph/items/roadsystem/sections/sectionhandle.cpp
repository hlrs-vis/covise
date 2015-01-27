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

#include "sectionhandle.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/roadsection.hpp"
#include "src/data/commands/roadsectioncommands.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/roadsystemitem.hpp"
#include "src/graph/items/roadsystem/roaditem.hpp"
#include "sectionhandle.hpp"
#include "sectionitem.hpp"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>

//################//
// CONSTRUCTOR    //
//################//

SectionHandle::SectionHandle(SectionItem *parentSectionItem)
    : QGraphicsPathItem(parentSectionItem)
    , parentRoadSystemItem_(NULL)
    , parentSectionItem_(parentSectionItem)
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

SectionHandle::SectionHandle(RoadSystemItem *parentRoadSystemItem)
    : QGraphicsPathItem(parentRoadSystemItem)
    , parentRoadSystemItem_(parentRoadSystemItem)
    , parentSectionItem_(NULL)
    , moveItem_(false)
{
    // Mode 2: Parent is a RoadSystemItem //
    //
    // The RoadSystemItem has one SectionHandle that is used for adding new sections.

    // Init //
    //
    init();
}

//################//
// FUNCTIONS      //
//################//

void
SectionHandle::init()
{
    // Create if necessary //
    //
    if (!SectionHandle::pathTemplate_)
    {
        SectionHandle::createPath();
    }

    // ZValue //
    //
    setZValue(1.0); // stack before siblings

    // Set path //
    //
    setPath(*pathTemplate_);
}

void
SectionHandle::updateTransform()
{
    bool tmp = moveItem_;
    moveItem_ = false; // set false so itemChange doesn't interrupt
    RoadSection *roadSection = parentSectionItem_->getRoadSection();
    setPos(roadSection->getParentRoad()->getGlobalPoint(roadSection->getSStart()));
    setRotation(roadSection->getParentRoad()->getGlobalHeading(roadSection->getSStart()));
    moveItem_ = tmp; // reset
}

void
SectionHandle::updatePos(RoadItem *roadItem, const QPointF &position, double sStartHint, double sEndHint)
{
    RSystemElementRoad *road = roadItem->getRoad();

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
SectionHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //

    // Mode 1 //
    //
    if (parentSectionItem_)
    {
        if (change == QGraphicsItem::ItemPositionChange)
        {
            if (moveItem_)
            {
                RoadSection *roadSection = parentSectionItem_->getRoadSection();
                double s = roadSection->getParentRoad()->getSFromGlobalPoint(value.toPointF(), roadSection->getSStart() - 100.0, roadSection->getSEnd());
                //				return parentSectionItem_->getRoadSection()->getParentRoad()->getGlobalPoint(s, 0.0);

                MoveRoadSectionCommand *command = new MoveRoadSectionCommand(roadSection, s, parentSectionItem_->getRoadSectionType());
                if (command->isValid())
                {
                    roadSection->getUndoStack()->push(command);
                }
                else
                {
                    delete command;
                }

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
SectionHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::ClosedHandCursor);
    moveItem_ = true;
    QGraphicsPathItem::mousePressEvent(event); // pass to baseclass
}

void
SectionHandle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
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
QPainterPath *SectionHandle::pathTemplate_ = NULL;

/*!
* Initializes the path (only once).
*/
void
SectionHandle::createPath()
{
    // Check first //
    //
    if (SectionHandle::pathTemplate_)
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

    SectionHandle::pathTemplate_ = &pathTemplate;
}
