/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   17.03.2010
**
**************************************************************************/

#ifndef SECTIONITEM_HPP
#define SECTIONITEM_HPP

#include "src/graph/items/graphelement.hpp"

class RoadItem;
class SectionHandle;

#include "src/data/roadsystem/rsystemelementroad.hpp"
class RoadSection;

class SectionItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SectionItem(RoadItem *parentRoadItem, RoadSection *roadSection);
    virtual ~SectionItem();

    virtual RSystemElementRoad::DRoadSectionType getRoadSectionType() const = 0; // To be defined by subclasses

    RoadItem *getParentRoadItem() const
    {
        return parentRoadItem_;
    }

    // Section //
    //
    RoadSection *getRoadSection() const
    {
        return roadSection_;
    }

    // ContextMenu //
    //
    QAction *getHideSectionAction() const
    {
        return hideSectionAction_;
    }
    QAction *getParentRoadAction() const
    {
        return hideParentRoadAction_;
    }
    QAction *getRemoveSectionAction() const
    {
        return removeSectionAction_;
    }
    QAction *getRemoveParentRoadAction() const
    {
        return removeParentRoadAction_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();

private:
    SectionItem(); /* not allowed */
    SectionItem(const SectionItem &); /* not allowed */
    SectionItem &operator=(const SectionItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual void hideParentRoad();
    virtual void hideRoads();
    virtual bool removeSection() = 0;
    virtual void removeParentRoad();

    //################//
    // EVENTS         //
    //################//

protected:
    //################//
    // PROPERTIES     //
    //################//

protected:
    // Road //
    //
    RoadItem *parentRoadItem_;

    // Items //
    //
    SectionHandle *sectionHandle_;

private:
    // Section //
    //
    RoadSection *roadSection_;
    RSystemElementRoad *road_;

    // ContextMenu //
    //
    QAction *hideSectionAction_;
    QAction *hideParentRoadAction_;
    QAction *removeSectionAction_;
    QAction *removeParentRoadAction_;
};

#endif // SECTIONITEM_HPP
