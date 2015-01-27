/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.06.2010
**
**************************************************************************/

#ifndef ELEVATIONSECTIONPOLYNOMIALITEM_HPP
#define ELEVATIONSECTIONPOLYNOMIALITEM_HPP

#include "src/graph/items/roadsystem/sections/sectionitem.hpp"

class ElevationRoadPolynomialItem;

#include "src/data/roadsystem/rsystemelementroad.hpp"
class ElevationSection;

class ElevationSectionPolynomialItem : public SectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationSectionPolynomialItem(ElevationRoadPolynomialItem *parentRoadItem, ElevationSection *elevationSection);
    virtual ~ElevationSectionPolynomialItem();

    // RoadSectionType //
    //
    virtual RSystemElementRoad::DRoadSectionType getRoadSectionType() const
    {
        return RSystemElementRoad::DRS_ElevationSection;
    }

    // Section //
    //
    ElevationSection *getElevationSection() const
    {
        return elevationSection_;
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
    ElevationSectionPolynomialItem(); /* not allowed */
    ElevationSectionPolynomialItem(const ElevationSectionPolynomialItem &); /* not allowed */
    ElevationSectionPolynomialItem &operator=(const ElevationSectionPolynomialItem &); /* not allowed */

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
    ElevationRoadPolynomialItem *parentRoadItem_;

    // Section //
    //
    ElevationSection *elevationSection_;

    // ContextMenu //
    //
    QAction *splitAction_;
};

#endif // ELEVATIONSECTIONPOLYNOMIALITEM_HPP
