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

#ifndef ROADLINKITEM_HPP
#define ROADLINKITEM_HPP

#include "src/graph/items/graphelement.hpp"

#include "src/data/roadsystem/roadlink.hpp"

class RoadLinkRoadItem;
class RoadLinkHandle;

class RoadLinkItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadLinkItem(RoadLinkRoadItem *parent, RoadLink *roadLink);
    explicit RoadLinkItem(RoadLinkRoadItem *parent, RoadLink::RoadLinkType roadLinkType);
    virtual ~RoadLinkItem();

    // Road //
    //
    RSystemElementRoad *getParentRoad() const
    {
        return parentRoad_;
    }
    RoadLink *getRoadLink() const
    {
        return roadLink_;
    }
    RoadLink::RoadLinkType getRoadLinkType() const;

    // Graphics //
    //
    void updateTransform();
    void updateColor();
    virtual void createPath();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

private:
    RoadLinkItem(); /* not allowed */
    RoadLinkItem(const RoadLinkItem &); /* not allowed */
    RoadLinkItem &operator=(const RoadLinkItem &); /* not allowed */

    void init();

    void updateType();
    void updateParentRoad();
    void updatePathList();

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // RoadLink //
    //
    RoadLink *roadLink_;
    RoadLink::RoadLinkType type_;

    // Road //
    //
    RoadLinkRoadItem *parentRoadItem_;
    RSystemElementRoad *parentRoad_;

    // Linked elements //
    //
    RSystemElementRoad *targetRoad_;
    RSystemElementJunction *targetJunction_;
    QList<RSystemElementRoad *> paths_;

    // Handle //
    //
    RoadLinkHandle *roadLinkHandle_;
};

#endif // ROADLINKITEM_HPP
