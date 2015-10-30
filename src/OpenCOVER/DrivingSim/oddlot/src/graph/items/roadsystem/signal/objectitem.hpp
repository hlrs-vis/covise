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

#ifndef OBJECTITEM_HPP
#define OBJECTITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class ObjectObject;
class RoadSystemItem;
class ObjectTextItem;
class SignalEditor;
class SignalManager;

class QColor;

class ObjectItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ObjectItem(RoadSystemItem *roadSystemItem, Object *object, QPointF pos);
    virtual ~ObjectItem();

    // Garbage //
    //
    virtual bool deleteRequest();

    Object *getObject() const
    {
        return object_;
    }

    // Graphics //
    //
    void updateColor();
    virtual void createPath();
    void updatePosition();
	void updateCategory();

    // Text //
    //
    ObjectTextItem *getObjectTextItem() const
    {
        return objectTextItem_;
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

    bool removeObject();

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

    Object *object_;
    QPointF pos_;

    ObjectTextItem *objectTextItem_;

    QColor outerColor_;
	int categorySize_;

    SignalEditor *signalEditor_;

	SignalManager *signalManager_;


};

#endif // ROADITEM_HPP
