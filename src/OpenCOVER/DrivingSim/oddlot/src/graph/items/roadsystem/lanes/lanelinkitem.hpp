/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/29/2010
**
**************************************************************************/

#ifndef LANELINKITEM_HPP
#define LANELINKITEM_HPP

#include "src/graph/items/graphelement.hpp"

class LaneItem;
class LinkHandle;
#include "src/data/roadsystem/sections/lane.hpp"

class LaneLinkItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneLinkItem(LaneItem *parentLaneItem, Lane::D_LaneLinkType linkType);
    virtual ~LaneLinkItem();

    // Tools //
    //

    // Graphics //
    //
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
    LaneLinkItem(); /* not allowed */
    LaneLinkItem(const LaneLinkItem &); /* not allowed */
    LaneLinkItem &operator=(const LaneLinkItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // EVENTS         //
    //################//

protected:
    //	virtual void			mousePressEvent(QGraphicsSceneMouseEvent * event);
    //	virtual void			mouseReleaseEvent(QGraphicsSceneMouseEvent * event);
    //	virtual void			mouseMoveEvent(QGraphicsSceneMouseEvent * event);

    //################//
    // PROPERTIES     //
    //################//

private:
    LaneItem *laneItem_;
    Lane::D_LaneLinkType linkType_;

    Lane *lane_;
    LaneSection *laneSection_;
    RSystemElementRoad *road_;

    LaneSection *linkedLaneSection_;
    RSystemElementRoad *linkedRoad_;

    LinkHandle *linkHandle_;
};

#endif // LANELINKITEM_HPP
