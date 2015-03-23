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

#ifndef CONTROLLERITEM_HPP
#define CONTROLLERITEM_HPP

#include "src/graph/items/graphelement.hpp"

class RSystemElementController;
class RoadSystemItem;
class TextHandle;
class SignalEditor;

class ControllerItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ControllerItem(RoadSystemItem *roadSystemItem, RSystemElementController *controller);
    virtual ~ControllerItem();

    // Road //
    //
    RSystemElementController *getController() const
    {
        return controller_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();

private:
    ControllerItem(); /* not allowed */
    ControllerItem(const ControllerItem &); /* not allowed */
    ControllerItem &operator=(const ControllerItem &); /* not allowed */

    void init();
    void updatePath();

    //################//
    // SLOTS          //
    //################//

public slots:
    bool removeController();
    void addToCurrentTile();

    //################//
    // EVENTS         //
    //################//
    void mousePressEvent(QGraphicsSceneMouseEvent *event);

 protected:
    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);



public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Controller/Roads //
    //
    RSystemElementController *controller_;

    // RoadSystem //
    //
    RoadSystemItem *roadSystemItem_;

    // SignalEditor //
    //
    SignalEditor *signalEditor_;

    // Text //
    //
    TextHandle *textHandle_;
};

#endif // CONTROLLERITEM_HPP
