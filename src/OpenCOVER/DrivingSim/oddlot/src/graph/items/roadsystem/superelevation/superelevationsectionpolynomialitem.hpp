/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   16.07.2010
**
**************************************************************************/

#ifndef SUPERELEVATIONSECTIONPOLYNOMIALITEM_HPP
#define SUPERELEVATIONSECTIONPOLYNOMIALITEM_HPP

#include "src/graph/items/roadsystem/sections/sectionitem.hpp"

class SuperelevationRoadPolynomialItem;

#include "src/data/roadsystem/rsystemelementroad.hpp"
class SuperelevationSection;

class SuperelevationSectionPolynomialItem : public SectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SuperelevationSectionPolynomialItem(SuperelevationRoadPolynomialItem *parentRoadItem, SuperelevationSection *superelevationSection);
    virtual ~SuperelevationSectionPolynomialItem();

    // RoadSectionType //
    //
    virtual RSystemElementRoad::DRoadSectionType getRoadSectionType() const
    {
        return RSystemElementRoad::DRS_SuperelevationSection;
    }

    // Section //
    //
    SuperelevationSection *getSuperelevationSection() const
    {
        return superelevationSection_;
    }

    // Graphics //
    //
    void updateColor();
    virtual void createPath();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    SuperelevationSectionPolynomialItem(); /* not allowed */
    SuperelevationSectionPolynomialItem(const SuperelevationSectionPolynomialItem &); /* not allowed */
    SuperelevationSectionPolynomialItem &operator=(const SuperelevationSectionPolynomialItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    bool removeSection();
    void splitSection();

    //################//
    // EVENTS         //
    //################//

protected:
    //################//
    // PROPERTIES     //
    //################//

private:
    // RoadItem //
    //
    SuperelevationRoadPolynomialItem *parentRoadItem_;

    // Section //
    //
    SuperelevationSection *superelevationSection_;

    // ContextMenu //
    //
    QAction *splitAction_;
};

#endif // SUPERELEVATIONSECTIONPOLYNOMIALITEM_HPP
