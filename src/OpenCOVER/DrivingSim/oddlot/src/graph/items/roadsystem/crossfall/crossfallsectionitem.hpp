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

#ifndef CROSSFALLSECTIONITEM_HPP
#define CROSSFALLSECTIONITEM_HPP

#include "src/graph/items/roadsystem/sections/sectionitem.hpp"

#include "src/data/roadsystem/sections/crossfallsection.hpp"

class CrossfallEditor;

class CrossfallSectionItem : public SectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrossfallSectionItem(CrossfallEditor *crossfallEditor, RoadItem *parentRoadItem, CrossfallSection *crossfallSection);
    virtual ~CrossfallSectionItem();

    virtual RSystemElementRoad::DRoadSectionType getRoadSectionType() const
    {
        return RSystemElementRoad::DRS_CrossfallSection;
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
    CrossfallSectionItem(); /* not allowed */
    CrossfallSectionItem(const CrossfallSectionItem &); /* not allowed */
    CrossfallSectionItem &operator=(const CrossfallSectionItem &); /* not allowed */

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
    CrossfallEditor *crossfallEditor_;

    CrossfallSection *crossfallSection_;
};

#endif // CROSSFALLSECTIONITEM_HPP
