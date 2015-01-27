/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   21.06.2010
**
**************************************************************************/

#ifndef ELEVATIONSECTIONITEM_HPP
#define ELEVATIONSECTIONITEM_HPP

#include "src/graph/items/roadsystem/sections/sectionitem.hpp"

#include "src/data/roadsystem/sections/elevationsection.hpp"

class ElevationEditor;
class ProjectEditor;

class ElevationSectionItem : public SectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationSectionItem(ElevationEditor *elevationEditor, RoadItem *parentRoadItem, ElevationSection *elevationSection);
    virtual ~ElevationSectionItem();

    virtual RSystemElementRoad::DRoadSectionType getRoadSectionType() const
    {
        return RSystemElementRoad::DRS_ElevationSection;
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

    ElevationSection *getElevationSection()
    {
        return elevationSection_;
    };

private:
    ElevationSectionItem(); /* not allowed */
    ElevationSectionItem(const ElevationSectionItem &); /* not allowed */
    ElevationSectionItem &operator=(const ElevationSectionItem &); /* not allowed */

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
    ElevationEditor *elevationEditor_;

    ElevationSection *elevationSection_;
};

#endif // ELEVATIONSECTIONITEM_HPP
