/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#ifndef ROADLINKROADITEM_HPP
#define ROADLINKROADITEM_HPP

#include "src/graph/items/roadsystem/roaditem.hpp"

class RSystemElementRoad;

class RoadSystemItem;
class RoadLinkItem;
class RoadLinkSinkItem;

class RoadLinkRoadItem : public RoadItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadLinkRoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~RoadLinkRoadItem();

    // Graphics //
    //
    void updateColor();
    virtual void createPath();

    // RoadLink //
    //
    RoadLinkItem *getPredecessorItem() const
    {
        return predecessor_;
    }
    RoadLinkItem *getSuccessorItem() const
    {
        return successor_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    RoadLinkRoadItem(); /* not allowed */
    RoadLinkRoadItem(const RoadLinkRoadItem &); /* not allowed */
    RoadLinkRoadItem &operator=(const RoadLinkRoadItem &); /* not allowed */

    void init();

    void updatePredecessor();
    void updateSuccessor();

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // PROPERTIES     //
    //################//

private:
    RoadLinkItem *predecessor_;
    RoadLinkItem *successor_;

    RoadLinkSinkItem *startSinkItem_;
    RoadLinkSinkItem *endSinkItem_;
};

#endif // ROADLINKROADITEM_HPP
