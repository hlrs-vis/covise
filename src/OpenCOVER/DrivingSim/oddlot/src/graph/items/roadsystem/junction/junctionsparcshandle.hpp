/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/26/2010
**
**************************************************************************/

#ifndef JUNCTIONSPARCSHANDLE_HPP
#define JUNCTIONSPARCSHANDLE_HPP

#include "src/graph/items/handles/sliderhandle.hpp"

class JunctionSpArcSItem;

class JunctionSparcsHandle : public SliderHandle
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionSparcsHandle(JunctionSpArcSItem *parentSpArcSItem);
    virtual ~JunctionSparcsHandle();

    void setFactor(double factor);

protected:
private:
    JunctionSparcsHandle(); /* not allowed */
    JunctionSparcsHandle(const JunctionSparcsHandle &); /* not allowed */
    JunctionSparcsHandle &operator=(const JunctionSparcsHandle &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

protected:
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // SLOTS          //
    //################//

public slots:
    void moveHandlePositionChange(const QPointF &pos);

    //################//
    // PROPERTIES     //
    //################//

private:
private:
    JunctionSpArcSItem *parentSpArcSItem_;

    // Slider Dimensions //
    //
    double min_;
    double max_;
};

#endif // JUNCTIONSPARCSHANDLE_HPP
