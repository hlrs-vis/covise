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

#include "handle.hpp"

//################//
// CONSTRUCTOR    //
//################//

Handle::Handle(QGraphicsItem *parent, bool flip)
    : QObject()
    , QGraphicsPathItem(parent)
    , Observer()
{
    // ContextMenu //
    //
    contextMenu_ = new QMenu();

    // Flags //
    //
    setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
    setFlag(QGraphicsItem::ItemIgnoresParentOpacity, true);
    setAcceptHoverEvents(true);

    // Transformation //
    //
    // Note: The y-Axis flip is done here, because the item ignores
    // the view's transformation
    if (flip)
    {
        QTransform trafo;
        trafo.rotate(180, Qt::XAxis);
        setTransform(trafo);
    }

    // ZValue //
    //
    setZValue(1.0); // stack before siblings
}

Handle::~Handle()
{
    // ContextMenu //
    //
    delete contextMenu_;
}

//################//
// EVENTS         //
//################//

void
Handle::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    contextMenu_->exec(event->screenPos());
}
