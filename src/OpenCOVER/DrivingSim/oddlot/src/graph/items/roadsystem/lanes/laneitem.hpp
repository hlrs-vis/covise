/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/18/2010
**
**************************************************************************/

#ifndef LANEITEM_HPP
#define LANEITEM_HPP

#include "src/graph/items/graphelement.hpp"

class LaneSectionItem;

class LaneItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneItem(LaneSectionItem *parentLaneSectionItem, Lane *lane);
    virtual ~LaneItem();

    // Lane //
    //
    LaneSectionItem *getParentLaneSectionItem() const
    {
        return parentLaneSectionItem_;
    }
    Lane *getLane() const
    {
        return lane_;
    }

    // Graphics //
    //
    void updateColor();
    virtual void createPath();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();

private:
    LaneItem(); /* not allowed */
    LaneItem(const LaneItem &); /* not allowed */
    LaneItem &operator=(const LaneItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual void hideParentRoad();
    bool removeLane();
    virtual void removeParentRoad();

    //################//
    // EVENTS         //
    //################//

protected:
    //	virtual void			mousePressEvent(QGraphicsSceneMouseEvent * event);
    virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    //	virtual void			mouseMoveEvent(QGraphicsSceneMouseEvent * event);

    //################//
    // PROPERTIES     //
    //################//

private:
    LaneSectionItem *parentLaneSectionItem_;

    LaneSection *parentLaneSection_;
    Lane *lane_;

    RSystemElementRoad *grandparentRoad_;
};

#endif // LANEITEM_HPP
