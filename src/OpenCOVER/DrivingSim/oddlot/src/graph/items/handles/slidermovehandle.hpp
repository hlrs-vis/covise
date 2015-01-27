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

#ifndef SLIDERMOVEHANDLE_HPP
#define SLIDERMOVEHANDLE_HPP

#include "src/graph/items/handles/movehandle.hpp"

class SliderMoveHandle : public MoveHandle
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SliderMoveHandle(QGraphicsItem *parent);
    virtual ~SliderMoveHandle();

    void updateColor();

    // Observer Pattern //
    //
    //	virtual void			updateObserver();

protected:
private:
    SliderMoveHandle(); /* not allowed */
    SliderMoveHandle(const SliderMoveHandle &); /* not allowed */
    SliderMoveHandle &operator=(const SliderMoveHandle &); /* not allowed */

//################//
// SIGNALS        //
//################//

signals:
    void requestPositionChange(const QPointF &pos);

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
};

#endif // SLIDERMOVEHANDLE_HPP
