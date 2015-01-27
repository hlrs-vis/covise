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

#ifndef LANEWIDTHITEM_HPP
#define LANEWIDTHITEM_HPP

#include "src/graph/items/graphelement.hpp"

class LaneSectionWidthItem;
class LaneWidth;

class LaneWidthItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneWidthItem(LaneSectionWidthItem *parentLaneSectionWidthItem, LaneWidth *laneWidth);
    virtual ~LaneWidthItem();

    // Lane //
    //
    Lane *getParentLane() const
    {
        return parentLane_;
    }

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
    LaneWidthItem(); /* not allowed */
    LaneWidthItem(const LaneWidthItem &); /* not allowed */
    LaneWidthItem &operator=(const LaneWidthItem &); /* not allowed */

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
    LaneSectionWidthItem *parentLaneSectionWidthItem_;
    LaneWidth *laneWidth_;
    Lane *parentLane_;
};

#endif // LANEWIDTHITEM_HPP
