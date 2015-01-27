/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/25/2010
**
**************************************************************************/

#ifndef ROADTEXTITEM_HPP
#define ROADTEXTITEM_HPP

#include "src/graph/items/graphelement.hpp"

class RoadItem;
class RSystemElementRoad;
class TextHandle;

class RoadTextItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadTextItem(RoadItem *roadItem);
    virtual ~RoadTextItem();

    virtual void createPath();
    virtual QPainterPath shape() const;

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

private:
    RoadTextItem(); /* not allowed */
    RoadTextItem(const RoadTextItem &); /* not allowed */
    RoadTextItem &operator=(const RoadTextItem &); /* not allowed */

    void updatePosition();
    void updateName();
    RoadItem *getRoadItem()
    {
        return roadItem_;
    };

    //################//
    // SLOTS          //
    //################//

public slots:
    void handlePositionChange(const QPointF &pos);
    //	void						handleSelectionChange(bool selected);

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    //	virtual void			mouseReleaseEvent(QGraphicsSceneMouseEvent * event);
    //	virtual void			mouseMoveEvent(QGraphicsSceneMouseEvent * event);

    //	virtual void			hoverEnterEvent(QGraphicsSceneHoverEvent * event);
    //	virtual void			hoverLeaveEvent(QGraphicsSceneHoverEvent * event);
    //	virtual void			hoverMoveEvent(QGraphicsSceneHoverEvent * event);

    //################//
    // PROPERTIES     //
    //################//

private:
    RoadItem *roadItem_;
    RSystemElementRoad *road_;
    TextHandle *textHandle_;
};
#endif // ROADTEXTITEM_HPP
