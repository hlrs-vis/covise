/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   07.07.2010
**
**************************************************************************/

#ifndef ROADMOVEHANDLE_HPP
#define ROADMOVEHANDLE_HPP

#include "src/graph/items/handles/movehandle.hpp"

class TrackEditor;
class RSystemElementRoad;
class TrackComponent;

class RoadMoveHandle : public MoveHandle
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadMoveHandle(TrackEditor *trackEditor, RSystemElementRoad *road, QGraphicsItem *parent);
    virtual ~RoadMoveHandle();

    RSystemElementRoad *getRoad() const
    {
        return road_;
    }

    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
private:
    RoadMoveHandle(); /* not allowed */
    RoadMoveHandle(const RoadMoveHandle &); /* not allowed */
    RoadMoveHandle &operator=(const RoadMoveHandle &); /* not allowed */

    void updateColor();
    void updateTransform();

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    //	virtual void			mousePressEvent(QGraphicsSceneMouseEvent * event);
    //	virtual void			mouseReleaseEvent(QGraphicsSceneMouseEvent * event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

    //	virtual void			hoverEnterEvent(QGraphicsSceneHoverEvent * event);
    //	virtual void			hoverLeaveEvent(QGraphicsSceneHoverEvent * event);
    //	virtual void			hoverMoveEvent(QGraphicsSceneHoverEvent * event);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    TrackEditor *trackEditor_;

    RSystemElementRoad *road_;
    TrackComponent *firstTrack_;
};

#endif // ROADMOVEHANDLE_HPP
