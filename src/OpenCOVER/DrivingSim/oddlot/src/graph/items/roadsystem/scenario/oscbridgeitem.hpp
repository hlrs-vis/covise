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

#ifndef OSCBRIDGEITEM_HPP
#define OSCBRIDGEITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class Bridge;
class Tunnel;
class RoadSystemItem;
class BridgeTextItem;

class QColor;

class OSCBridgeItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OSCBridgeItem(RoadSystemItem *roadSystemItem, Bridge *bridge, QPointF pos);
    virtual ~OSCBridgeItem();

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
    BridgeTextItem *getBridgeTextItem() const
    {
        return bridgeTextItem_;
    }


    //################//
    // EVENTS         //
    //################//

protected:
    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
    void init();

	RoadSystemItem *roadSystemItem_;
    Bridge *bridge_;
	Tunnel *tunnel_;
	RSystemElementRoad *road_;
    QPointF pos_;

	QPainterPath *path_;

    BridgeTextItem *bridgeTextItem_;

    QColor outerColor_;
};

#endif // OSCBRIDGEITEM_HPP
