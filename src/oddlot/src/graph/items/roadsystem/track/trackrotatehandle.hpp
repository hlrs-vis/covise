/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   27.04.2010
**
**************************************************************************/

#ifndef TRACKROTATEHANDLE_HPP
#define TRACKROTATEHANDLE_HPP

#include "src/graph/items/handles/rotatehandle.hpp"

#include <QPointF>

class TrackEditor;
class TrackComponent;

class TrackRotateHandle : public RotateHandle
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackRotateHandle(TrackEditor *trackEditor, QGraphicsItem *parent);
    virtual ~TrackRotateHandle();

    void registerLowSlot(TrackComponent *trackComponent);
    void registerHighSlot(TrackComponent *trackComponent);

    int getRotDOF() const
    {
        return rotDOF_;
    }
    TrackComponent *getLowSlot() const
    {
        return lowSlot_;
    }
    TrackComponent *getHighSlot() const
    {
        return highSlot_;
    }

    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
private:
    TrackRotateHandle(); /* not allowed */
    TrackRotateHandle(const TrackRotateHandle &); /* not allowed */

    void updateDOF();
    int calculateRotDOF();
    void updateColor();

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

    TrackComponent *lowSlot_;
    TrackComponent *highSlot_;

    int rotDOF_;

    // Mouse Move Pos //
    //
    QPointF mousePos_;
};

#endif // TRACKROTATEHANDLE_HPP
