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

#ifndef ROADMARKLANESECTIONITEM_HPP
#define ROADMARKLANESECTIONITEM_HPP

#include "src/graph/items/roadsystem/sections/sectionitem.hpp"

class RoadMarkLaneSectionItem : public SectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadMarkLaneSectionItem(RoadItem *parentRoadItem, LaneSection *laneSection);
    virtual ~RoadMarkLaneSectionItem();

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

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

private:
    RoadMarkLaneSectionItem(); /* not allowed */
    RoadMarkLaneSectionItem(const RoadMarkLaneSectionItem &); /* not allowed */
    RoadMarkLaneSectionItem &operator=(const RoadMarkLaneSectionItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:

    virtual bool removeSection()
    {
        return false;
    };

    //################//
    // PROPERTIES     //
    //################//

private:
    // LaneSection //
    //
    LaneSection *laneSection_;
};

#endif // ROADMARKLANESECTIONITEM_HPP
