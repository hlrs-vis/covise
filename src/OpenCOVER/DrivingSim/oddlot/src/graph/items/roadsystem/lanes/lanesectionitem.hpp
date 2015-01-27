/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/15/2010
**
**************************************************************************/

#ifndef LANESECTIONITEM_HPP
#define LANESECTIONITEM_HPP

#include "src/graph/items/roadsystem/sections/sectionitem.hpp"

class LaneEditor;
class ProjectEditor;

class LaneSectionItem : public SectionItem
{
    Q_OBJECT
    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneSectionItem(LaneEditor *laneEditor, RoadItem *parentRoadItem, LaneSection *laneSection);
    virtual ~LaneSectionItem();

    virtual RSystemElementRoad::DRoadSectionType getRoadSectionType() const
    {
        return RSystemElementRoad::DRS_LaneSection;
    }

    // LaneSection //
    //
    LaneSection *getLaneSection() const
    {
        return laneSection_;
    }
    LaneEditor *getLaneEditor() const
    {
        return laneEditor_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();

private:
    LaneSectionItem(); /* not allowed */
    LaneSectionItem(const LaneSectionItem &); /* not allowed */
    LaneSectionItem &operator=(const LaneSectionItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual bool removeSection();

    //################//
    // EVENTS         //
    //################//

public:
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
    // LaneSection //
    //
    LaneSection *laneSection_;
    LaneEditor *laneEditor_;
};

#endif // LANESECTIONITEM_HPP
