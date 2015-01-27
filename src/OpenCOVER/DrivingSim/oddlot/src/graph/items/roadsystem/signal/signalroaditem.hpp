/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#ifndef SIGNALROADITEM_HPP
#define SIGNALROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;
class RoadTextItem;
class SignalEditor;

class SignalRoadItem : public RoadItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~SignalRoadItem();

    // Graphics //
    //
    void updateColor();
    virtual void createPath();

    // Garbage //
    //
    //	virtual void			notifyDeletion();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    //################//
    // SLOTS          //
    //################//

public slots:

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
    // PROPERTIES     //
    //################//

private:
    void init();

    RoadSystemItem *roadSystemItem_;

    RSystemElementRoad *road_;
    SignalEditor *signalEditor_;

    QPointF pos_;
};

#endif // ROADITEM_HPP
