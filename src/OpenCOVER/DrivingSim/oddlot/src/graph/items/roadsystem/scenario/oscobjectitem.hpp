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

#ifndef OSCOBJECTITEM_HPP
#define OSCOBJECTITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class ObjectObject;
class RoadSystemItem;
class ObjectTextItem;
class SignalEditor;
class SignalManager;

class QColor;

class OSCObjectItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OSCObjectItem(RoadSystemItem *roadSystemItem, Object *object, QPointF pos);
    virtual ~OSCObjectItem();

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

    // Text //
    //
    ObjectTextItem *getObjectTextItem() const
    {
        return objectTextItem_;
    }

    // Garbage //
    //
    //	virtual void			notifyDeletion();


    //################//
    // SLOTS          //
    //################//

public slots:

    bool removeObject();

    //################//
    // EVENTS         //
    //################//

public:

protected:
    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
    void init();

	RoadSystemItem * roadSystemItem_;

    Object *object_;
	RSystemElementRoad *road_;
    QPointF pos_;
	QPainterPath *path_;

	bool doPan_;
	bool copyPan_;

    ObjectTextItem *objectTextItem_;

    QColor outerColor_;

};

#endif // OSCOBJECTITEM_HPP
