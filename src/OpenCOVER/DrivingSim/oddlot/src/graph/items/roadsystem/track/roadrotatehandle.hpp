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

#ifndef ROADROTATEHANDLE_HPP
#define ROADROTATEHANDLE_HPP

#include "src/graph/items/handles/rotatehandle.hpp"

#include <QPointF>

class TrackEditor;

class RSystemElementRoad;
class TrackComponent;

class RoadRotateHandle : public RotateHandle
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadRotateHandle(TrackEditor *trackEditor, RSystemElementRoad *road, QGraphicsItem *parent);
    virtual ~RoadRotateHandle();

    RSystemElementRoad *getRoad() const
    {
        return road_;
    }

    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
private:
    RoadRotateHandle(); /* not allowed */
    RoadRotateHandle(const RoadRotateHandle &); /* not allowed */
    RoadRotateHandle &operator=(const RoadRotateHandle &); /* not allowed */

    void updateColor();
    void updateTransform();

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

	virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    TrackEditor *trackEditor_;

    RSystemElementRoad *road_;
    TrackComponent *firstTrack_;

    // Mouse Move Pos //
    //
    QPointF mousePos_;
};

#endif // ROADROTATEHANDLE_HPP
