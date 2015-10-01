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

#ifndef SIGNALITEM_HPP
#define SIGNALITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class SignalObject;
class RoadSystemItem;
class SignalTextItem;
class SignalEditor;
class SignalManager;

class QColor;

class SignalItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalItem(RoadSystemItem *roadSystemItem, Signal *signal, QPointF pos);
    virtual ~SignalItem();

    // Garbage //
    //
    virtual bool deleteRequest();

    Signal *getSignal() const
    {
        return signal_;
    }

    // Graphics //
    //
    void updateColor();
    virtual void createPath();
    void updatePosition();

    // Text //
    //
    SignalTextItem *getSignalTextItem() const
    {
        return signalTextItem_;
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

    bool removeSignal();

    //################//
    // EVENTS         //
    //################//

public:
    //	virtual QVariant		itemChange(GraphicsItemChange change, const QVariant & value);
    void mousePressEvent(QGraphicsSceneMouseEvent *event);

protected:
    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
    void init();

    Signal *signal_;
    QPointF pos_;

    SignalTextItem *signalTextItem_;

    QColor outerColor_;

    SignalEditor *signalEditor_;
    
	SignalManager *signalManager_;

	int categorySize_;
};

#endif // ROADITEM_HPP
