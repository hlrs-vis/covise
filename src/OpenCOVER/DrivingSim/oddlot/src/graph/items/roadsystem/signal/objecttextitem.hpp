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

#ifndef OBJECTTEXTITEM_HPP
#define OBJECTEXTITEM_HPP

#include "src/graph/items/graphelement.hpp"

class ObjectItem;
class Object;
class TextHandle;

class ObjectTextItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ObjectTextItem(ObjectItem *objectItem);
    virtual ~ObjectTextItem();

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
    ObjectTextItem(); /* not allowed */
    ObjectTextItem(const ObjectTextItem &); /* not allowed */
    ObjectTextItem &operator=(const ObjectTextItem &); /* not allowed */

    void updatePosition();
    void updateName();
    ObjectItem *getObjectItem()
    {
        return objectItem_;
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
    ObjectItem *objectItem_;
    Object *object_;
    TextHandle *textHandle_;
};
#endif // ObjectTextItem_HPP
