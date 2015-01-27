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

#ifndef LANESECTIONWIDTHITEM_HPP
#define LANESECTIONWIDTHITEM_HPP

#include "src/graph/items/graphelement.hpp"

class LaneEditor;
class LaneSectionItem;
class LaneWidthRoadSystemItem;

class LaneSectionWidthItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneSectionWidthItem(LaneWidthRoadSystemItem *parent, Lane *lane);
    virtual ~LaneSectionWidthItem();

    // Lane //
    //
    //LaneSectionItem *		getParentLaneSectionItem() const { return parentLaneSectionItem_; }
    Lane *getLane() const
    {
        return lane_;
    }

    // Graphics (zero- line) //
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
    LaneSectionWidthItem(); /* not allowed */
    LaneSectionWidthItem(const LaneSectionWidthItem &); /* not allowed */
    LaneSectionWidthItem &operator=(const LaneSectionWidthItem &); /* not allowed */

    void init();

    // Handles //
    //
    void rebuildMoveHandles();
    void deleteMoveHandles();
    //################//
    // SLOTS          //
    //################//

public slots:

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
    // Handles //
    //
    QGraphicsPathItem *moveHandles_;

    //LaneSectionItem *		parentLaneSectionItem_;

    LaneEditor *laneEditor_;
    LaneSection *parentLaneSection_;
    Lane *lane_;

    RSystemElementRoad *grandparentRoad_;
};

#endif // LANEITEM_HPP
