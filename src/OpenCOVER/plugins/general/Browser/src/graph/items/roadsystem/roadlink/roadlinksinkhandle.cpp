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

#include "roadlinksinkhandle.hpp"


 // Graph //
 //
#include "src/graph/projectgraph.hpp"
#include "src/graph/topviewgraph.hpp"
#include "src/graph/editors/roadlinkeditor.hpp"

#include "roadlinksinkitem.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Qt //
//
#include <QCursor>
#include <QGraphicsSceneMouseEvent>

//################//
// CONSTRUCTOR    //
//################//

RoadLinkSinkHandle::RoadLinkSinkHandle(RoadLinkSinkItem *parentRoadLinkSinkItem, RoadLinkEditor *editor)
    : CircularHandle(parentRoadLinkSinkItem)
    , parentRoadLinkSinkItem_(parentRoadLinkSinkItem)
    , editor_(editor)
{

    // Flags //
    //
    setFlag(QGraphicsItem::ItemIsSelectable, true);
}

RoadLinkSinkHandle::~RoadLinkSinkHandle()
{
}

//################//
// EVENTS         //
//################//

void
RoadLinkSinkHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ODD::ToolId tool = editor_->getCurrentTool();

    if (!isSelected())
    {
        if (tool == ODD::TRL_SINK)
        {
            if (!editor_->registerLinkSinkHandle(this, parentRoadLinkSinkItem_->getParentRoad()))
                event->ignore();
        }
        else if (editor_->getCurrentParameterTool() == ODD::TPARAM_SELECT)
        {
            event->ignore();
        }
    }
    else
    {
        if ((tool == ODD::TRL_LINK) || (tool == ODD::TRL_SINK))
        {
            editor_->deregisterHandle(this, ODD::TRL_SINK);
        }
    }
}


void
RoadLinkSinkHandle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::OpenHandCursor);
    Handle::hoverEnterEvent(event);
}

void
RoadLinkSinkHandle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
    Handle::hoverLeaveEvent(event);
}

void
RoadLinkSinkHandle::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    Handle::hoverMoveEvent(event);
}
