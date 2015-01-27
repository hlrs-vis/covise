/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   27.03.2010
**
**************************************************************************/

#ifndef TYPESECTIONITEM_HPP
#define TYPESECTIONITEM_HPP

#include "src/graph/items/roadsystem/sections/sectionitem.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"

class RoadTypeEditor;
class ProjectEditor;

class TypeSectionItem : public SectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TypeSectionItem(RoadTypeEditor *typeEditor, RoadItem *parentRoadItem, TypeSection *typeSection);
    virtual ~TypeSectionItem();

    virtual RSystemElementRoad::DRoadSectionType getRoadSectionType() const
    {
        return RSystemElementRoad::DRS_TypeSection;
    }

    // Tools //
    //
    void changeRoadType(TypeSection::RoadType roadType);

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
    TypeSectionItem(); /* not allowed */
    TypeSectionItem(const TypeSectionItem &); /* not allowed */
    TypeSectionItem &operator=(const TypeSectionItem &); /* not allowed */

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
    RoadTypeEditor *typeEditor_;

    TypeSection *typeSection_;
};

#endif // TYPESECTIONITEM_HPP
