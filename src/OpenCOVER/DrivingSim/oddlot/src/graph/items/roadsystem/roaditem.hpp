/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#ifndef ROADITEM_HPP
#define ROADITEM_HPP

#include "src/graph/items/graphelement.hpp"

class RSystemElementRoad;
class RoadSystemItem;
class RoadTextItem;

class RoadItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadItem(RoadSystemItem *roadSystemItem, RSystemElementRoad *road);
    virtual ~RoadItem();

    // Road //
    //
    RSystemElementRoad *getRoad() const
    {
        return road_;
    }

    // Text //
    //
    RoadTextItem *getTextItem() const
    {
        return textItem_;
    }

    // Garbage //
    //
    virtual void notifyDeletion(); // to be implemented by subclasses

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();

private:
    RoadItem(); /* not allowed */
    RoadItem(const RoadItem &); /* not allowed */
    RoadItem &operator=(const RoadItem &); /* not allowed */

    void init();

    //################//
    // SLOTS          //
    //################//

public slots:
    bool removeRoad();
    void hideRoad();
    bool removeRoadLink();
    void addToCurrentTile();

    //################//
    // EVENTS         //
    //################//

public:
    //	virtual QVariant		itemChange(GraphicsItemChange change, const QVariant & value);
    void mousePressEvent(QGraphicsSceneMouseEvent *event);

protected:
    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Road //
    //
    RSystemElementRoad *road_;

    // RoadSystem //
    //
    RoadSystemItem *roadSystemItem_;

    // Text //
    //
    RoadTextItem *textItem_;
};

#endif // ROADITEM_HPP
