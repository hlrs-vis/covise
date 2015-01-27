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

#ifndef TEXTHANDLE_HPP
#define TEXTHANDLE_HPP

#include "handle.hpp"

class TextHandle : public Handle
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TextHandle(const QString &text, QGraphicsItem *parent);
    virtual ~TextHandle();

    QString getText() const;
    void setText(const QString &text);

    void setRoadPos(const QPointF &pos);

protected:
    virtual void createPath();

private:
    TextHandle(); /* not allowed */
    TextHandle(const TextHandle &); /* not allowed */
    TextHandle &operator=(const TextHandle &); /* not allowed */

//################//
// SIGNALS        //
//################//

signals:
    void requestPositionChange(const QPointF &pos);
    void selectionChanged(bool selected);

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
    QGraphicsTextItem textItem_;

    QPointF roadPos_;
};

#endif // TEXTHANDLE_HPP
