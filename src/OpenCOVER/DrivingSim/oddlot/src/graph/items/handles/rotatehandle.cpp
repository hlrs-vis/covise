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

#include "rotatehandle.hpp"

//################//
// CONSTRUCTOR    //
//################//

RotateHandle::RotateHandle(QGraphicsItem *parent)
    : Handle(parent)
{
    // Path //
    //
    if (!RotateHandle::pathTemplate_)
    {
        createPath();
    }
    setPath(*RotateHandle::pathTemplate_);
}

RotateHandle::~RotateHandle()
{
    //	xy->detachObserver(this);
}

//################//
// EVENTS         //
//################//

void
RotateHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::ClosedHandCursor);
    Handle::mousePressEvent(event); // pass to baseclass
}

void
RotateHandle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::OpenHandCursor);
    Handle::mouseReleaseEvent(event); // pass to baseclass
}

void
RotateHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    Handle::mouseMoveEvent(event); // pass to baseclass
}

void
RotateHandle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::OpenHandCursor);
    Handle::hoverEnterEvent(event);
}

void
RotateHandle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
    Handle::hoverLeaveEvent(event);
}

void
RotateHandle::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    Handle::hoverMoveEvent(event);
}

//################//
// STATIC         //
//################//

/*! \brief Initialize the path once.
*
*/
void
RotateHandle::createPath()
{
    static QPainterPath pathTemplate; // Static, so the destructor kills it on application shutdown.

    double size = 1.5 * 4.0;
    double length = 5.0 * 4.0;
    pathTemplate.moveTo(length, 0.0);
    //	pathTemplate.lineTo(-length, 0.0);
    pathTemplate.lineTo(0.0, 0.0);
    //	pathTemplate.addEllipse(-length -size/2.0, -size/2.0, size, size);
    pathTemplate.addEllipse(length - size / 2.0, -size / 2.0, size, size);

    pathTemplate_ = &pathTemplate;
}

// Initialize to NULL //
//
QPainterPath *RotateHandle::pathTemplate_ = NULL;
