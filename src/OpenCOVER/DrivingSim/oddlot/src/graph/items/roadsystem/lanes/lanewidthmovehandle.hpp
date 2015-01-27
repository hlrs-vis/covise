/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   25.06.2010
**
**************************************************************************/

#ifndef LANEWIDTHMOVEHANDLE_HPP
#define LANEWIDTHMOVEHANDLE_HPP

#include "src/graph/items/handles/movehandle.hpp"

class LaneEditor;
class LaneWidth;

class LaneWidthMoveHandle : public MoveHandle
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneWidthMoveHandle(LaneEditor *laneEditor, QGraphicsItem *parent);
    virtual ~LaneWidthMoveHandle();

    void registerLowSlot(LaneWidth *laneWidthSection);
    void registerHighSlot(LaneWidth *laneWidthSection);

    LaneWidth *getLowSlot() const
    {
        return lowSlot_;
    }
    LaneWidth *getHighSlot() const
    {
        return highSlot_;
    }

    int getPosDOF() const
    {
        return posDOF_;
    }
    void setDOF(int dof);

    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
private:
    LaneWidthMoveHandle(); /* not allowed */
    LaneWidthMoveHandle(const LaneWidthMoveHandle &); /* not allowed */
    LaneWidthMoveHandle &operator=(const LaneWidthMoveHandle &); /* not allowed */

    void updateColor();

    //################//
    // SLOTS          //
    //################//

public slots:
    void removeCorner();
    void smoothCorner();

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    LaneEditor *laneEditor_;

    LaneWidth *lowSlot_;
    LaneWidth *highSlot_;

    int posDOF_;

    // ContextMenu //
    //
    QAction *removeAction_;
    QAction *smoothAction_;
};

#endif // LANEWIDTHMOVEHANDLE_HPP
