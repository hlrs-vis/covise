/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#ifndef SHAPESECTIONITEM_HPP
#define SHAPESECTIONITEM_HPP

#include "src/graph/items/roadsystem/sections/sectionitem.hpp"


class ShapeSection;
class ShapeEditor;
class ShapeSectionPolynomialItems;

class ShapeSectionItem : public SectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeSectionItem(ShapeEditor *shapeEditor, RoadItem *parentRoadItem, ShapeSection *shapeSection);
    virtual ~ShapeSectionItem();

    virtual RSystemElementRoad::DRoadSectionType getRoadSectionType() const
    {
        return RSystemElementRoad::DRS_ShapeSection;
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
    ShapeSectionItem(); /* not allowed */
    ShapeSectionItem(const ShapeSectionItem &); /* not allowed */
    ShapeSectionItem &operator=(const ShapeSectionItem &); /* not allowed */

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
    ShapeEditor *shapeEditor_;

    ShapeSection *shapeSection_;

	ShapeSectionPolynomialItems *shapeSectionPolynomialItems_;
};

#endif // SHAPESECTIONITEM_HPP
