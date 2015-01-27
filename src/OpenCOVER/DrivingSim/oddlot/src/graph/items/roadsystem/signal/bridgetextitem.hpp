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

#ifndef BRIDGETEXTITEM_HPP
#define BRIDGETEXTITEM_HPP

#include "src/graph/items/graphelement.hpp"

class BridgeItem;
class Bridge;
class TextHandle;

class BridgeTextItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit BridgeTextItem(BridgeItem *bridgeItem);
    virtual ~BridgeTextItem();

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
    BridgeTextItem(); /* not allowed */
    BridgeTextItem(const BridgeTextItem &); /* not allowed */
    BridgeTextItem &operator=(const BridgeTextItem &); /* not allowed */

    void updatePosition();
    void updateName();
    BridgeItem *getBridgeItem()
    {
        return bridgeItem_;
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
    BridgeItem *bridgeItem_;
    Bridge *bridge_;
    TextHandle *textHandle_;
};
#endif // BridgeTextItem_HPP
