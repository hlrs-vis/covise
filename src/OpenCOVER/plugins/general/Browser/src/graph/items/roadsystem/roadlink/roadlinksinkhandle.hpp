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

#ifndef ROADLINKSINKHANDLE_HPP
#define ROADLINKSINKHANDLE_HPP

#include "src/graph/items/handles/circularhandle.hpp"

class RoadLinkSinkItem;
class RoadLinkEditor;

class RoadLinkSinkHandle : public CircularHandle
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadLinkSinkHandle(RoadLinkSinkItem *parentRoadLinkItem, RoadLinkEditor *editor);
    virtual ~RoadLinkSinkHandle();

    RoadLinkSinkItem *getParentRoadLinkSinkItem() const
    {
        return parentRoadLinkSinkItem_;
    }

    void updateTransformation();
    void updateColor();

protected:
private:
    RoadLinkSinkHandle(); /* not allowed */
    RoadLinkSinkHandle(const RoadLinkSinkHandle &); /* not allowed */
    RoadLinkSinkHandle &operator=(const RoadLinkSinkHandle &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

protected:
//    virtual QVariant itemChange(GraphicsItemChange change, const QVariant & value);

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);

    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
    RoadLinkSinkItem *parentRoadLinkSinkItem_;
    RoadLinkEditor *editor_;
};

#endif // ROADLINKSINKHANDLE_HPP
