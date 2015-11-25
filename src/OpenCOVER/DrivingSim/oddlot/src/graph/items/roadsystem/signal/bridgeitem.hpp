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

#ifndef BRIDGEITEM_HPP
#define BRIDGEITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class Bridge;
class Tunnel;
class RoadSystemItem;
class BridgeTextItem;
class SignalEditor;
class SignalManager;

class QColor;

class BridgeItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit BridgeItem(RoadSystemItem *roadSystemItem, Bridge *bridge, QPointF pos);
    virtual ~BridgeItem();

    // Garbage //
    //
    virtual bool deleteRequest();

    Bridge *getBridge() const
    {
        return bridge_;
    }

	Tunnel *getTunnel() const
    {
        return tunnel_;
    }

    // Graphics //
    //
    void updateColor();
    virtual void createPath();
    void updatePosition();

    // Text //
    //
    BridgeTextItem *getBridgeTextItem() const
    {
        return bridgeTextItem_;
    }

    // Garbage //
    //
    //	virtual void			notifyDeletion();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    //################//
    // SLOTS          //
    //################//

public slots:

    bool removeBridge();

    //################//
    // EVENTS         //
    //################//

public:
    //	virtual QVariant		itemChange(GraphicsItemChange change, const QVariant & value);
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
	void keyPressEvent(QKeyEvent *event);
	void keyReleaseEvent(QKeyEvent *event);

protected:
    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
    void init();

	RoadSystemItem *roadSystemItem_;
    Bridge *bridge_;
	Tunnel *tunnel_;
	RSystemElementRoad *road_;
    QPointF pos_;

	QPointF pressPos_;
	QPointF lastPos_;
	bool doPan_;
	bool copyPan_;

	QPainterPath *path_;

	RSystemElementRoad *closestRoad_;

    BridgeTextItem *bridgeTextItem_;

    QColor outerColor_;
	int categorySize_;

    SignalEditor *signalEditor_;

	SignalManager *signalManager_;
};

#endif // ROADITEM_HPP
