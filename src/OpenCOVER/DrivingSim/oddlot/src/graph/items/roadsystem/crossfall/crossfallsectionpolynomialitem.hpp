/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.07.2010
**
**************************************************************************/

#ifndef CROSSFALLSECTIONPOLYNOMIALITEM_HPP
#define CROSSFALLSECTIONPOLYNOMIALITEM_HPP

#include "src/graph/items/roadsystem/sections/sectionitem.hpp"

class CrossfallRoadPolynomialItem;

#include "src/data/roadsystem/rsystemelementroad.hpp"
class CrossfallSection;

class CrossfallSectionPolynomialItem : public SectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrossfallSectionPolynomialItem(CrossfallRoadPolynomialItem *parentRoadItem, CrossfallSection *crossfallSection);
    virtual ~CrossfallSectionPolynomialItem();

    // RoadSectionType //
    //
    virtual RSystemElementRoad::DRoadSectionType getRoadSectionType() const
    {
        return RSystemElementRoad::DRS_CrossfallSection;
    }

    // Section //
    //
    CrossfallSection *getCrossfallSection() const
    {
        return crossfallSection_;
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
    CrossfallSectionPolynomialItem(); /* not allowed */
    CrossfallSectionPolynomialItem(const CrossfallSectionPolynomialItem &); /* not allowed */
    CrossfallSectionPolynomialItem &operator=(const CrossfallSectionPolynomialItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual bool removeSection();
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
    CrossfallRoadPolynomialItem *parentRoadItem_;

    // Section //
    //
    CrossfallSection *crossfallSection_;

    // ContextMenu //
    //
    QAction *splitAction_;
};

#endif // CROSSFALLSECTIONPOLYNOMIALITEM_HPP
