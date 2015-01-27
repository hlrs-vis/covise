/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/27/2010
**
**************************************************************************/

#include "slidermovehandle.hpp"

//################//
// CONSTRUCTOR    //
//################//

SliderMoveHandle::SliderMoveHandle(QGraphicsItem *parent)
    : MoveHandle(parent)
{
    // Color/Transform //
    //
    updateColor();
}

SliderMoveHandle::~SliderMoveHandle()
{
}

//################//
// FUNCTIONS      //
//################//

void
SliderMoveHandle::updateColor()
{
    setBrush(QBrush(ODD::instance()->colors()->brightGreen()));
    setPen(QPen(ODD::instance()->colors()->darkGreen()));
}

//################//
// EVENTS         //
//################//

/*! \brief Handles the item's position changes.
*/
QVariant
SliderMoveHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //

    if (change == QGraphicsItem::ItemPositionChange)
    {
        return pos(); // no translation
    }

    return MoveHandle::itemChange(change, value);
}

void
SliderMoveHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::ClosedHandCursor);
    MoveHandle::mousePressEvent(event); // pass to baseclass
}

void
SliderMoveHandle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::OpenHandCursor);
    MoveHandle::mouseReleaseEvent(event); // pass to baseclass
}

void
SliderMoveHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    // Let the trackEditor handle the movement //
    //
    if (isSelected()) // only if this handle is actually selected
    {
        emit requestPositionChange(event->pos());
        return; // do not apply translation yourself!
        //		trackEditor_->translateSliderMoveHandles(scenePos(), event->scenePos());
    }

    MoveHandle::mouseMoveEvent(event); // pass to baseclass
}
