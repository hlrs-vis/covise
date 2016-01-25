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

#ifndef OSCSIGNALITEM_HPP
#define OSCSIGNALITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class SignalObject;
class RoadSystemItem;
class SignalTextItem;

class QColor;

class OSCSignalItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OSCSignalItem(RoadSystemItem *roadSystemItem, Signal *signal, QPointF pos);
    virtual ~OSCSignalItem();

    // Garbage //
    //
	virtual bool deleteRequest(){return false;};

    // Graphics //
    //
    void updateColor();
    virtual void createPath();
    void updatePosition();

    // Text //
    //
    SignalTextItem *getSignalTextItem() const
    {
        return signalTextItem_;
    }

    //################//
    // EVENTS         //
    //################//

public:

protected:
    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
	RoadSystemItem * roadSystemItem_;
    void init();

    Signal *signal_;
	RSystemElementRoad *road_;

    QPointF pos_;

    SignalTextItem *signalTextItem_;

    QColor outerColor_;
};

#endif // OSCSIGNALITEM_HPP
