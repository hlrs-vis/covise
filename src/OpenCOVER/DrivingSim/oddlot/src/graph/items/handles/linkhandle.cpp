/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10.05.2010
**
**************************************************************************/

#include "linkhandle.hpp"

//################//
// CONSTRUCTOR    //
//################//

LinkHandle::LinkHandle(QGraphicsItem *parent)
    : Handle(parent)
    , linkHandleType_(DHLT_START)
{
    // Path //
    //
    if (!LinkHandle::pathTemplate_)
    {
        createPath();
    }
    setPath(*LinkHandle::pathTemplate_);
}

LinkHandle::~LinkHandle()
{
}

//################//
// FUNCTIONS      //
//################//

void
LinkHandle::setHandleType(LinkHandle::LinkHandleType linkHandleType)
{
    linkHandleType_ = linkHandleType;

    setPath(*LinkHandle::pathTemplate_);

    if (linkHandleType == LinkHandle::DHLT_END)
    {
        QPainterPath thePath = path();
        thePath.translate(-halfheight_ * 1.414213, 0.0);
        setPath(thePath);
    }

    if (linkHandleType == LinkHandle::DHLT_CENTER)
    {
        QPainterPath thePath = path();
        thePath.translate(-0.5 * halfheight_ * 1.414213, 0.0);
        setPath(thePath);
    }
}

//################//
// OBSERVER       //
//################//

//################//
// EVENTS         //
//################//

void
LinkHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::ClosedHandCursor);
    Handle::mousePressEvent(event); // pass to baseclass
}

void
LinkHandle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::OpenHandCursor);
    Handle::mouseReleaseEvent(event); // pass to baseclass
}

void
LinkHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    Handle::mouseMoveEvent(event); // pass to baseclass
}

void
LinkHandle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::OpenHandCursor);
    Handle::hoverEnterEvent(event); // pass to baseclass
}

void
LinkHandle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
    Handle::hoverLeaveEvent(event); // pass to baseclass
}

void
LinkHandle::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    Handle::hoverMoveEvent(event); // pass to baseclass
}

//################//
// STATIC         //
//################//

/*! \brief Initialize the path once.
*
*/
void
LinkHandle::createPath()
{
    static QPainterPath pathTemplate; // Static, so the destructor kills it on application shutdown.

    //	double size = 2.0*4.0;
    //	pathTemplate.moveTo(-1.0, -size/2.0);
    //	pathTemplate.lineTo(-1.0, size/2.0);
    //	pathTemplate.lineTo(-1.0+1.414213*size/2.0, 0.0);
    //	pathTemplate.lineTo(-1.0, -size/2.0);

    double width = halfheight_ * 1.414213; // hard coded above
    pathTemplate.moveTo(0.0, -halfheight_);
    pathTemplate.lineTo(0.0, halfheight_);
    pathTemplate.lineTo(width, 0.0);
    pathTemplate.lineTo(0.0, -halfheight_);

    pathTemplate_ = &pathTemplate;
}

// Initialize to NULL //
//
QPainterPath *LinkHandle::pathTemplate_ = NULL;

double LinkHandle::halfheight_ = 1.5 * 4.0;
