/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/18/2010
**
**************************************************************************/

#ifndef ROADLINKHANDLE_HPP
#define ROADLINKHANDLE_HPP

#include "src/graph/items/handles/linkhandle.hpp"

class RoadLinkItem;

class RoadLinkHandle : public LinkHandle
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadLinkHandle(RoadLinkItem *parentRoadLinkItem);
    virtual ~RoadLinkHandle();

    RoadLinkItem *getParentRoadLinkItem() const
    {
        return parentRoadLinkItem_;
    }

    void updateTransformation();
    void updateColor();

protected:
private:
    RoadLinkHandle(); /* not allowed */
    RoadLinkHandle(const RoadLinkHandle &); /* not allowed */
    RoadLinkHandle &operator=(const RoadLinkHandle &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

protected:
    //	virtual QVariant		itemChange(GraphicsItemChange change, const QVariant & value);

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
    RoadLinkItem *parentRoadLinkItem_;
};

#endif // ROADLINKHANDLE_HPP
