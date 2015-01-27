/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/18/2010
**
**************************************************************************/

#include "roadlinkhandle.hpp"

// Data //
//
//#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
//#include "src/data/commands/trackcommands.hpp"

// Graph //
//
#include "src/graph/projectgraph.hpp"

#include "roadlinkitem.hpp"

// Qt //
//
#include <QCursor>
#include <QGraphicsSceneMouseEvent>

//################//
// CONSTRUCTOR    //
//################//

RoadLinkHandle::RoadLinkHandle(RoadLinkItem *parentRoadLinkItem)
    : LinkHandle(parentRoadLinkItem)
    , parentRoadLinkItem_(parentRoadLinkItem)
{
    setHandleType(LinkHandle::DHLT_CENTER);

    // Flags //
    //
    setFlag(QGraphicsItem::ItemIsSelectable, true);
}

RoadLinkHandle::~RoadLinkHandle()
{
}

//################//
// EVENTS         //
//################//

void
RoadLinkHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::ClosedHandCursor);
    Handle::mousePressEvent(event); // pass to baseclass
}

void
RoadLinkHandle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::OpenHandCursor);
    Handle::mouseReleaseEvent(event); // pass to baseclass
}

void
RoadLinkHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    Handle::mouseMoveEvent(event); // pass to baseclass
}

void
RoadLinkHandle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::OpenHandCursor);
    Handle::hoverEnterEvent(event);
}

void
RoadLinkHandle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
    Handle::hoverLeaveEvent(event);
}

void
RoadLinkHandle::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    Handle::hoverMoveEvent(event);
}
