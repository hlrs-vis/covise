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

#ifndef OSCSIGNALROADITEM_HPP
#define OSCSIGNALROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;
class RoadSystemItem;
class RoadTextItem;

class OSCRoadItem : public RoadItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OSCRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~OSCRoadItem();

	// Garbage //
    //
	virtual bool deleteRequest(){return false;};

    // Graphics //
    //
    void updateColor();
    virtual void createPath();

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // EVENTS         //
    //################//

protected:

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

    QPointF pos_;
};

#endif // OSCROADITEM_HPP
