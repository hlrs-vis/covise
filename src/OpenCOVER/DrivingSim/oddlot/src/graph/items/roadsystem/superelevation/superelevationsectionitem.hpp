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

#ifndef SUPERELEVATIONSECTIONITEM_HPP
#define SUPERELEVATIONSECTIONITEM_HPP

#include "src/graph/items/roadsystem/sections/sectionitem.hpp"

#include "src/data/roadsystem/sections/superelevationsection.hpp"

class SuperelevationEditor;

class SuperelevationSectionItem : public SectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SuperelevationSectionItem(SuperelevationEditor *superelevationEditor, RoadItem *parentRoadItem, SuperelevationSection *superelevationSection);
    virtual ~SuperelevationSectionItem();

    virtual RSystemElementRoad::DRoadSectionType getRoadSectionType() const
    {
        return RSystemElementRoad::DRS_SuperelevationSection;
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
    SuperelevationSectionItem(); /* not allowed */
    SuperelevationSectionItem(const SuperelevationSectionItem &); /* not allowed */
    SuperelevationSectionItem &operator=(const SuperelevationSectionItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual bool removeSection();

    //################//
    // EVENTS         //
    //################//

protected:
    //	virtual QVariant		itemChange(GraphicsItemChange change, const QVariant & value);

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
    SuperelevationEditor *superelevationEditor_;

    SuperelevationSection *superelevationSection_;
};

#endif // SUPERELEVATIONSECTIONITEM_HPP
