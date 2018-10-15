/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/24/2010
**
**************************************************************************/

#include "texthandle.hpp"

//################//
// CONSTRUCTOR    //
//################//

TextHandle::TextHandle(const QString &text, QGraphicsItem *parent, bool flip)
    : Handle(parent, flip)
    , textItem_(text, this)
{
    // Flags //
    //
    setFlag(QGraphicsItem::ItemIsMovable, true);

    textItem_.setPos(10.0, 6.0);

    createPath();
}

TextHandle::~TextHandle()
{
}

//################//
// FUNCTIONS      //
//################//

QString
TextHandle::getText() const
{
    return textItem_.toPlainText();
}

void
TextHandle::setText(const QString &text)
{
    textItem_.setPlainText(text);
    createPath();
}

void
TextHandle::createPath()
{
    QPainterPath thePath;

    double width = textItem_.boundingRect().width();
    double height = 16.0;
    thePath.addRect(10.0, 10.0, width, height);

    setPath(thePath);
}

//################//
// OBSERVER       //
//################//

//################//
// EVENTS         //
//################//

QVariant
TextHandle::itemChange(GraphicsItemChange change, const QVariant &value)
{
    // NOTE: position is relative to parent!!! //
    //

    if (change == QGraphicsItem::ItemPositionChange)
    {
        return pos(); // no translation
    }

    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        emit selectionChanged(value.toBool()); // handle has to be set selectable first
    }

    return Handle::itemChange(change, value);
}

void
TextHandle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        setCursor(Qt::ClosedHandCursor);
    }

    Handle::mousePressEvent(event); // pass to baseclass
}

void
TextHandle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        setCursor(Qt::OpenHandCursor);
    }

    Handle::mouseReleaseEvent(event); // pass to baseclass
}

void
TextHandle::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    // Let someone else handle the movement //
    //
    //	if(isSelected()) // only if this handle is actually selected
    {
        emit requestPositionChange(event->scenePos() - event->lastScenePos());
        return; // do not apply translation yourself!
    }

    Handle::mouseMoveEvent(event); // pass to baseclass
}

void
TextHandle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::OpenHandCursor);
    Handle::hoverEnterEvent(event); // pass to baseclass
}

void
TextHandle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
    Handle::hoverLeaveEvent(event); // pass to baseclass
}

void
TextHandle::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    Handle::hoverMoveEvent(event); // pass to baseclass
}
