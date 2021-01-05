/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/22/2010
**
**************************************************************************/

#ifndef ROADMARKITEM_HPP
#define ROADMARKITEM_HPP

#include "src/graph/items/graphelement.hpp"

class RoadMarkLaneItem;
#include "src/data/roadsystem/sections/laneroadmark.hpp"

class RoadMarkItem : public GraphElement
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadMarkItem(RoadMarkLaneItem *parentLaneItem, LaneRoadMark *roadMark);
    virtual ~RoadMarkItem();

    // Graphics //
    //
    void updateColor();
    virtual void createPath();
    void updatePen();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

    RSystemElementRoad *getParentRoad()
    {
        return parentRoad_;
    };

private:
    RoadMarkItem(); /* not allowed */
    RoadMarkItem(const RoadMarkItem &); /* not allowed */
    RoadMarkItem &operator=(const RoadMarkItem &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

protected:
    virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
    RoadMarkLaneItem *parentLaneItem_;

    Lane *parentLane_;
    LaneSection *parentLaneSection_;
    RSystemElementRoad *parentRoad_;

    LaneRoadMark *roadMark_;

    LaneRoadMark::RoadMarkType lastRoadMarkType_;
};

#endif // ROADMARKITEM_HPP
