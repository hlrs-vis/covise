/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   27.05.2010
**
**************************************************************************/

#include "movehandle.hpp"

//################//
// CONSTRUCTOR    //
//################//

MoveHandle::MoveHandle(QGraphicsItem *parent)
    : Handle(parent)
{
    // Path //
    //
    if (!MoveHandle::pathTemplate_)
    {
        createPath();
    }
    setPath(*MoveHandle::pathTemplate_);

    // Flags //
    //
    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
}

MoveHandle::~MoveHandle()
{
    //	xy->detachObserver(this);
}

//################//
// EVENTS         //
//################//

void
MoveHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::ClosedHandCursor);
    Handle::mousePressEvent(event); // pass to baseclass
}

void
MoveHandle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::CrossCursor);
    Handle::mouseReleaseEvent(event); // pass to baseclass
}

void
MoveHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    Handle::mouseMoveEvent(event); // pass to baseclass
}

void
MoveHandle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::CrossCursor);
    Handle::hoverEnterEvent(event);
}

void
MoveHandle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
    Handle::hoverLeaveEvent(event);
}

void
MoveHandle::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    Handle::hoverMoveEvent(event);
}

//################//
// EVENTS         //
//################//

///*! \brief Handles the item's position changes.
//*/
//QVariant
//	MoveHandle
//	::itemChange(GraphicsItemChange change, const QVariant & value)
//{
//	return Handle::itemChange(change, value);
//}

//################//
// STATIC         //
//################//

/*! \brief Initialize the path once.
*
*/
void
MoveHandle::createPath()
{
    static QPainterPath pathTemplate; // Static, so the destructor kills it on application shutdown.

    double size = 2.0 * 4.0;
    pathTemplate.addRect(-size / 2.0, -size / 2.0, size, size);

    pathTemplate_ = &pathTemplate;
}

// Initialize to NULL //
//
QPainterPath *MoveHandle::pathTemplate_ = NULL;
